#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import os
import time
import numpy as np
import astropy.io.fits as pyfits

from lsst.utils import getPackageDir
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.meas.base as meaBase
import lsst.meas.algorithms as meaAlg
import lsst.meas.extensions.shapeHSM
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask,CatalogCalculationTask
from fpfsBase import fpfsBaseTask

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class g3CatalogConfig(pexConfig.Config):
    "config"
    doGetIni =   pexConfig.Field(dtype=bool, default=False, doc = "whether to add the intrinsic snr to output catalog")
    doCoNoi =   pexConfig.Field(dtype=bool, default=True, doc = "remove the noise correlation?")
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Detect sources"
    )
    deblend = pexConfig.ConfigurableField(
        target = SourceDeblendTask,
        doc = "Split blended source into their components"
    )
    measurement = pexConfig.ConfigurableField(
        target = SingleFrameMeasurementTask,
        doc = "Measure sources"
    )
    catalogCalculation = pexConfig.ConfigurableField(
        target = CatalogCalculationTask,
        doc = "Subtask to run catalogCalculation plugins on catalog"
    )
    fpfsBase = pexConfig.ConfigurableField(
        target = fpfsBaseTask,
        doc = "Subtask to run measurement of fps method"
    )
    
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.detection.thresholdValue = 5.0
        self.detection.isotropicGrow  = True
        self.detection.reEstimateBackground=True
        self.deblend.propagateAllPeaks = True 
        self.measurement.plugins.names = [
            "base_SdssCentroid",
            'base_GaussianFlux',
            "base_SdssShape",
            "base_PsfFlux",
            "base_Blendedness",
            ]
        self.load(os.path.join(getPackageDir("obs_subaru"), "config", "cmodel.py"))
        self.measurement.load(os.path.join(getPackageDir("obs_subaru"), "config", "hsm.py"))
        self.measurement.slots.apFlux       =   None
        self.measurement.slots.instFlux     =   None
        self.measurement.slots.calibFlux    =   None

class g3CatalogTask(pipeBase.CmdLineTask):
    ConfigClass = g3CatalogConfig
    _DefaultName = "g3Catalog"
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        self.makeSubtask('fpfsBase', schema=self.schema)
        if self.config.doGetIni:
            self.schema.addField("snrI", type=np.float32, doc="galaxy's intrinsic snr")
            self.schema.addField("resI", type=np.float32, doc="galaxy's intrinsic resolution")
        self.schema.addField("CModel_mag", type=np.float32, doc="galaxy's measured resolution")
        self.schema.addField("detected", type=int, doc="wheter galaxy is detected by hscpipe")
        
    def addFootPrint(self,exposure,sources):
        for ss in sources:
            ss['detected']=1
        offset  =   afwGeom.Extent2I(10,10)
        dims    =   afwGeom.Extent2I(20, 20)
        mskData =   exposure.getMaskedImage().getMask().getArray()
        imageBBox=  exposure.getBBox()
        for j in range(100):
            for i in range(100):
                centx   =   i*64+32
                centy   =   j*64+32
                hasSource=  False
                for jj in range(centy-15,centy+15):
                    if hasSource:
                        break
                    for ii in range(centx-15,centx+15):
                        if mskData[jj,ii]>1:
                            hasSource=True
                            break
                if not hasSource:
                    record = sources.addNew()
                    position = afwGeom.Point2I(centx,centy)
                    bbox = afwGeom.Box2I(position-offset, dims)
                    footprint = afwDet.Footprint(bbox, imageBBox)
                    footprint.addPeak(position.getX(), position.getY(),1.)
                    record.setFootprint(footprint)
                    record['detected']=0
        return
    
    def loadPreOutcome(self,sources,inputdir,sName,ifile):
        datref  =   pyfits.getdata(inputdir+'%s/epoch_catalog-%04d-0.fits'%(sName,ifile))
        snrGrid =   np.zeros((100,100))
        resGrid =   np.zeros((100,100))
        for ssRef in datref:
            iG  =   int(ssRef['x']/64)
            jG  =   int(ssRef['y']/64)
            snrGrid[jG,iG]=ssRef['snr']
            resGrid[jG,iG]=ssRef['resolution']
        for ss in sources:
            iG=int(ss.getFootprint().getCentroid().getX()/64)
            jG=int(ss.getFootprint().getCentroid().getY()/64)
            ss['snrI']=snrGrid[jG,iG]
            ss['resI']=resGrid[jG,iG]
        return 
    
    def setPsfExp(self,exposure,psfImg):
        """
        load the PSF data
        """
        nx      = psfImg.getWidth()/3
        ny      = psfImg.getWidth()/3
        # extract only the lower-left image
        bbox    = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(nx, ny))
        psfImg2 = psfImg[bbox]
        psfData = psfImg2.getArray()[:,:]
        """
        setup PSF
        """
        ny=psfData.shape[0]-1
        nx=psfData.shape[1]-1
        psfImg=afwImg.ImageF(nx,ny)
        psfImg.getArray()[:,:]=psfData[0:ny,0:nx]
        psfImg = psfImg.convertD()
        # shift by half a pixel in both directions, so it's centered on a pixel
        psfImg = afwMath.offsetImage(psfImg, -0.5, -0.5)
        kernel = afwMath.FixedKernel(psfImg)
        kernelPSF=meaAlg.KernelPsf(kernel)
        exposure.setPsf(kernelPSF)
        return 

    def setVarExp(self,exposure):
        # get the variance of noise
        imgData =   exposure.getMaskedImage().getImage().getArray()[10:-10,10:-10]
        mskData =   exposure.getMaskedImage().getMask().getArray()[10:-10,10:-10]
        undetPix= imgData[mskData==0]
        var     =   np.std(undetPix, dtype=np.float64)**2.
        exposure.getMaskedImage().getVariance().getArray()[:,:]=var
        return

    def getCorNoise(self,corInput):
        # get the correlation of noise
        corImg  =   afwImg.ImageF(corInput)
        return corImg
    
    @pipeBase.timeMethod
    def run(self,ifile):
        print "running on the %04d exposure" %(ifile)
        np.random.seed(ifile)
        #ouptput file
        sName   =   'sample4'#'v2'#
        outputdir=  '/data2b/work/xiangchong.li/g3Test/%s-withMag3/' %(sName)
        catOutput=  outputdir+'src-%04d.fits'%(ifile)
        if os.path.exists(catOutput):
            print "Already have the outcome"
            return
        inputdir=   '/data2b/work/rmandelb/'#'/data3a/work/miyatake/hsc_sims/'#
        #load galaxy image
        ngrid   =   64
        galInput=   inputdir+'%s/image-%04d-0.fits'%(sName,ifile)
        if not os.path.exists(galInput):
            return
        expData =   pyfits.getdata(galInput)
        exposure=   afwImg.ExposureF(ngrid*100,ngrid*100)
        exposure.getMaskedImage().getImage().getArray()[:,:]=expData
        #prepare the wcs
        crval   =   afwCoord.IcrsCoord(0.*afwGeom.degrees, 0.*afwGeom.degrees)
        crpix   =   afwGeom.Point2D(0.0, 0.0)
        cdelt   =   (0.168*afwGeom.arcseconds).asDegrees()
        dataWcs =   afwImg.makeWcs(crval, crpix, cdelt, 0.0, 0.0, cdelt)
        exposure.setWcs(dataWcs)
        #prepare the frc
        dataCalib = afwImg.Calib()
        dataCalib.setFluxMag0(63095734448.0194)
        exposure.setCalib(dataCalib)
        # load PSF image
        psfInput=   inputdir+'%s/starfield_image-%04d-0.fits'%('sample3',ifile)
        psfImg  =   afwImg.ImageF(psfInput)
        self.setPsfExp(exposure,psfImg)
        # do detection        
        table = afwTable.SourceTable.make(self.schema)
        table.setMetadata(self.algMetadata)
        detRes  = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sources = detRes.sources
        # get the preMeasurements
        self.addFootPrint(exposure,sources)
        if self.config.doGetIni:
            self.loadPreOutcome(sources,inputdir,sName,ifile)
        # setup variance map
        self.setVarExp(exposure)
        # do deblending
        self.deblend.run(exposure=exposure, sources=sources)
        #sources=sources[123:2123]
        # do measurement
        self.measurement.run(measCat=sources, exposure=exposure)
        for ss in sources:
            if ss['modelfit_CModel_flux']>0:
                ss['CModel_mag']=dataCalib.getMagnitude(ss['modelfit_CModel_flux'])
        # measurement on the catalog level
        self.catalogCalculation.run(sources)
        # get the noise correlation
        corInput=   './correlation.fits'
        if self.config.doCoNoi and os.path.exists(corInput):
            corData =   self.getCorNoise(corInput)
        else:
            corData =   None
        print corData
        # run fps method
        time0   =   time.time()
        self.fpfsBase.run(sources,exposure,corData)
        time1   =   time.time()
        tperG   =   (time1-time0)/len(sources)
        print tperG
        print "writing %04d catalog" %(ifile)
        sources.writeFits(catOutput)
        return

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser

    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass

    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass

    def writeMetadata(self, dataRef):
        pass

    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass


class g3CatalogBatchConfig(pexConfig.Config):
    minField =   pexConfig.Field(dtype=int, default=0, doc = "minimum field number")
    maxField =   pexConfig.Field(dtype=int, default=1, doc = "maximum field number")
    g3Catalog = pexConfig.ConfigurableField(
        target = g3CatalogTask,
        doc = "g3Catalog task to run on multiple cores"
    )
    
class g3CatalogRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class g3CatalogBatchTask(BatchPoolTask):
    ConfigClass = g3CatalogBatchConfig
    RunnerClass = g3CatalogRunner
    _DefaultName = "g3CatalogBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("g3Catalog")
    
    @abortOnError
    def run(self,Id):
        #Prepare the pool
        pool    =   Pool("g3Catalog")
        pool.cacheClear()
        fieldList=  range(self.config.minField,self.config.maxField)
        pool.map(self.process,fieldList)
        return
        
    def process(self,cache,ifield):
        return self.g3Catalog.run(ifield)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser
    
    @classmethod
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass

    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    
    def writeMetadata(self, dataRef):
        pass
    
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    
    def _getConfigName(self):
        return None
   
    def _getEupsVersionsName(self):
        return None
    
    def _getMetadataName(self):
        return None
