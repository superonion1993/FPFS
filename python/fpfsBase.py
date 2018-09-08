#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
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
import numpy as np
import astropy.io.fits as pyfits
# for pipe task
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
# lsst.afw...
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
# noise replacer
from lsst.meas.base.noiseReplacer import NoiseReplacerConfig,NoiseReplacer

class fpfsBaseConfig(pexConfig.Config):
    "config"
    ngrid   =   pexConfig.Field(dtype=int, default=64, doc = "the stamp size")
    alpha   =   pexConfig.Field(dtype=float, default=4., doc = "cut-off ratio")
    beta    =   pexConfig.Field(dtype=float, default=.85, doc = "Shapelets scale ratio")
    dedge   =   pexConfig.Field(dtype=int, default=0, doc = "minimum distance to the edge")
    doPlot  =   pexConfig.Field(dtype=bool, default=False, doc = "plot the galaxy image?")
    

class fpfsBaseTask(pipeBase.CmdLineTask):
    ConfigClass = fpfsBaseConfig
    _DefaultName = "fpfsBase"
    def __init__(self,schema,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        schema.addField("fps_momentsG", type="ArrayD", doc="galaxy shapelets moments", size=6)
        schema.addField("fps_momentsN", type="ArrayD", doc="galaxy shapelets moments", size=6)
    
    def reScaleCor(self,exposure,corData0,ngrid):
        imgData =   exposure.getMaskedImage().getImage().getArray()[10:-10,10:-10]
        mskData =   exposure.getMaskedImage().getMask().getArray()[10:-10,10:-10]
        mask    =   (mskData==0)&(~np.isnan(imgData))
        undetPix=   imgData[mask]
        var     =   np.std(undetPix, dtype=np.float64)**2.
        # get the correlation of noise
        ngridI  =   corData0.getWidth()-1
        shift   =   (ngrid-ngridI)/2
        corData =   np.zeros((ngrid,ngrid))
        corData[shift:shift+ngridI,shift:shift+ngridI]  =   corData0.getArray()[0:ngridI,0:ngridI]*var
        return corData
    
    
    def getBadBits(self,mask):
        # get the bad Bits
        badBitsName=["BAD","SAT","INTRP","CR","NO_DATA","SUSPECT","UNMASKEDNAN","CROSSTALK",'NOT_DEBLENDED','DETECTED_NEGATIVE']
        badBits=0
        for item in badBitsName:
            try:
                badBits |= mask.getPlaneBitMask(item)
            except Exception:
                badBits  |=0
        return badBits
    
    def getPSF(self,psfExp,centroid,ngrid):
        psfData=np.zeros((ngrid,ngrid))
        try:
            psfData2=psfExp.computeImage(centroid).getArray() 
        except:
            return None
        shift   =   (ngrid-psfData2.shape[0]+1)/2
        for j in range(psfData2.shape[0]):
            for i in range(psfData2.shape[1]):
                psfData[shift+j,shift+i]=psfData2[j,i]
        return psfData
    
    def shapeletsPrepare(self,nord,ngrid,sigma):
        # prepare the shapelets function
        mord    =   nord
        rfunc   =   np.zeros((ngrid,ngrid))
        afunc   =   np.zeros((ngrid,ngrid))
        lfunc   =   np.zeros((nord+1,mord+1,ngrid,ngrid))
        chiC    =   np.zeros((nord+1,mord+1,ngrid,ngrid))
        chiS    =   np.zeros((nord+1,mord+1,ngrid,ngrid))
        #set up the r*r and theta function
        for j in range(ngrid):
            for i in range(ngrid):
                x   =   (i-ngrid/2.)/sigma
                y   =   (j-ngrid/2.)/sigma
                r   =   np.sqrt(x**2.+y**2.)
                rfunc[j,i]=r
                if r==0:
                    afunc[j,i]=0
                elif y>=0:
                    afunc[j,i]=np.arccos(x/r)
                elif y<0:
                    afunc[j,i]=-np.arccos(x/r)
        #set up l function
        for n in range(nord+1):
            for m in range(mord+1):
                if n==0:
                    lfunc[n,m,:,:]=1.
                elif n==1:
                    lfunc[n,m,:,:]=m+1.-rfunc*rfunc
                elif n>1:
                    lfunc[n,m,:,:]=(2.+(m-1.-rfunc*rfunc)/n)*lfunc[n-1,m,:,:]-(1.+(m-1.)/n)*lfunc[n-2,m,:,:]
        for nn in range(nord+1):
            for mm in range(nn,-1,-2):
                c1=(nn-abs(mm))/2
                d1=(nn+abs(mm))/2
                cc=np.math.factorial(c1)+0.
                dd=np.math.factorial(d1)+0.
                cc=cc/dd/np.pi
                chiC[nn,mm,:,:]=pow(-1.,d1)/sigma*pow(cc,0.5)*lfunc[c1,abs(mm),:,:]*pow(rfunc,abs(mm))*np.exp(-rfunc*rfunc/2.)*np.cos(mm*afunc)
                chiS[nn,mm,:,:]=pow(-1.,d1)/sigma*pow(cc,0.5)*lfunc[c1,abs(mm),:,:]*pow(rfunc,abs(mm))*np.exp(-rfunc*rfunc/2.)*np.sin(mm*afunc)
        return pipeBase.Struct(
            chiC=chiC,
            chiS=chiS
            )
    
    def getrCut(self,ss,alpha,ngrid):
        #determine the minR
        footprint   =   ss.getFootprint()
        minR        =   max(footprint.getBBox().getHeight(),footprint.getBBox().getWidth())/2.+3.
        #determin the radius
        sShape      =   ss.getShape()
        radius      =   max(sShape.getTraceRadius(),sShape.getDeterminantRadius())*alpha
        snr         =   ss['base_SdssShape_flux']/ss['base_SdssShape_fluxSigma']
        if np.isnan(radius) or snr<=5.:
            radius  =   minR
        radius      =   max(radius,minR)
        radius      =   max(radius,8)
        radius      =   min(radius,ngrid/2)
        return int(radius+0.5),minR
    
    def getNoiPow(self,ncorData,weightArray):
        ngrid  =   weightArray.shape[0]
        #get the weight array
        weightArray =   np.fft.fftshift(np.fft.ifft2(abs(np.fft.fft2(weightArray))**2.).real) 
        #Get the noise power
        npowData    =   np.zeros((ngrid,ngrid))
        for j in range(-9,9):
            for i in range(-9,9):
                npowData[ngrid/2+j,ngrid/2+i]=ncorData[ngrid/2+j,ngrid/2+i]*weightArray[ngrid/2+j,ngrid/2+i]
        npowData = np.fft.ifftshift(npowData)
        npowData = np.fft.fft2(npowData).real
        return npowData

    def getPow(self,galData,npowData=None):
        galpow  =   np.abs(np.fft.fft2(galData))**2.
        if npowData is not None:
            galpow  =   galpow-npowData
        galpow  =   np.fft.fftshift(galpow)
        return galpow
    
    def getHLRnaive(self,imgData,beta):
        ny = imgData.shape[0]
        nx = imgData.shape[1]
        thres=imgData.max()*0.5
        sigma=0.
        for j in range(ny):
            for i in range(nx):
                if imgData[j,i]>thres:
                    sigma+=1.
        sigma=np.sqrt(sigma/np.pi)*beta
        return sigma

    def removeStep(self,powData):
        ngrid=powData.shape[0]
        ave=0.
        num=0.
        for i in range(ngrid):
            ave+=powData[0,i]
            ave+=powData[ngrid-1,i]
            ave+=powData[i,0]
            ave+=powData[i,ngrid-1]
            num+=4.
        ave=ave/num
        powData2=   powData-ave*np.ones(powData.shape)
        return powData2

    def getGalDat(self,ss,exposure,psfExp,ngrid,ncorData,badBits,othBits,thisBits,alpha,hduList,dedge):
        expBox  =   exposure.getBBox()
        mi      =   exposure.getMaskedImage()
        im      =   mi.getImage()
        imDat   =   im.getArray()
        mskDat  =   mi.getMask().getArray()
        #get centroid
        fp      =   ss.getFootprint()
        bbox    =   fp.getBBox()
        pks     =   fp.getPeaks()
        if len(pks)==1:
            centroid=   pks[0].getCentroid()
        else:
            centroid=   fp.getCentroid()
        if not bbox.contains(afwGeom.Point2I(centroid)):
            print 'the centroid is not in the footprint'
            return None
        centx       =   int(centroid.getX()+0.5-expBox.getBeginX())
        centy       =   int(centroid.getY()+0.5-expBox.getBeginY())
        #get image        
        rCut,minR   =   self.getrCut(ss,alpha,ngrid)
        #rCut        =   ngrid/2
        left        =   centx-rCut
        right       =   centx+rCut
        down        =   centy-rCut
        up          =   centy+rCut
        #Test for edge
        isthrow  =  False
        isthrow |=  (left <   dedge)
        isthrow |=  (right>  imDat.shape[1]-dedge-1)
        isthrow |=  (down <   dedge)
        isthrow |=  (up   >  imDat.shape[0]-dedge-1)
        if isthrow:
            print 'the source is too close to the edge'
            return None
        # load information
        numBad      =   0
        numNei      =   0.
        mskData     =   np.zeros((ngrid,ngrid),dtype=int)
        galData     =   np.zeros((ngrid,ngrid))
        weightArray =   np.zeros((ngrid,ngrid))
        shift       =   ngrid/2-rCut
        for j in range(2*rCut):
            jc  =   shift+j-ngrid/2.
            for i in range(2*rCut):
                ic  =   shift+i-ngrid/2.
                r   =   np.sqrt(ic**2.+jc**2.)
                if r<rCut:
                    mskData[shift+j,shift+i]+=   1
                    galData[shift+j,shift+i]=   imDat[down+j,left+i]
                    weightArray[shift+j,shift+i]=   1.
                    if (mskDat[down+j,left+i]&othBits):
                        mskData[shift+j,shift+i]+=  2
                        numNei+=1.
                    if mskDat[down+j,left+i]&thisBits:
                        mskData[shift+j,shift+i]+=  4
                if mskDat[down+j,left+i]&badBits:
                    numBad+=1
        if numBad>=1:
            print 'too many bad pixels within the aperture'
            return None
        psfData     =   self.getPSF(psfExp,centroid,ngrid)  
        if psfData is None:
            print 'cannot determine the psf model'
            return None
        if ncorData is not None:
            npowData    =   self.getNoiPow(ncorData,weightArray)
        else:
            npowData    =   None
        psfData     =   self.getPow(psfData)
        galData     =   self.getPow(galData,npowData)
        galData     =   self.removeStep(galData)
        if hduList is not None:
            hduList.append(pyfits.ImageHDU(galData)) 
            hduList.append(pyfits.ImageHDU(psfData)) 
            hduList.append(pyfits.ImageHDU(npowData)) 
        return pipeBase.Struct(
            gal=galData,
            psf=psfData
            )
    
    def deconvolvePow(self,galData,psfData):
        height= galData.shape[0]
        width = galData.shape[1]
        thres = psfData.max()*1.e-4
        galDataU=np.zeros(galData.shape)
        for j in range(height):
            for i in range(width):
                if psfData[j,i]>=thres:
                    galDataU[j,i]=galData[j,i]/psfData[j,i]
        return galDataU
    
    def measMoments(self,data,momentsBase):
        height  =   data.shape[0]
        width   =   data.shape[1]
        chiC    =   momentsBase.chiC 
        chiS    =   momentsBase.chiS
        M00     =   0.
        M22c    =   0.
        M22s    =   0.
        M40     =   0.
        M42c    =   0.
        M42s    =   0.
        for j in range(height):
            for i in range(width):
                M00 +=data[j,i]*chiC[0,0,j,i]
                M22c+=data[j,i]*chiC[2,2,j,i]
                M22s+=data[j,i]*chiS[2,2,j,i]
                M40 +=data[j,i]*chiC[4,0,j,i]
                M42c+=data[j,i]*chiC[4,2,j,i]
                M42s+=data[j,i]*chiS[4,2,j,i]
        return np.array([M00,M22c,M22s,M40,M42c,M42s]) 
    
    def oneMeasure(self,ss,exposure,psfExp,corData,badBits,othBits,thisBits,hduList=None):
        ngrid   =   self.config.ngrid
        alpha   =   self.config.alpha
        beta    =   self.config.beta
        dedge   =   self.config.dedge
        GPNData =   self.getGalDat(ss,exposure,psfExp,ngrid,corData,badBits,othBits,thisBits,alpha,hduList,dedge)
        if GPNData is None:
            return
        galData =   GPNData.gal   
        psfData =   GPNData.psf   
        galDataU=   self.deconvolvePow(galData,psfData)
        #get PSF power and radius
        sigma    =   self.getHLRnaive(psfData,beta)
        #get the shapelets file
        shapeletsBase=self.shapeletsPrepare(4,ngrid,sigma)
        ss['fps_momentsG']=  self.measMoments(galDataU,shapeletsBase)
        return

    @pipeBase.timeMethod
    def run(self,sources,exposure,corData0=None):
        # load correlation image
        if corData0 is not None:
            corData =   self.reScaleCor(exposure,corData0,self.config.ngrid)
        else:
            corData =   None
        print corData
        # psfExp
        psfExp  =   exposure.getPsf()
        # noise replacement
        footprints = {measRecord.getId(): (measRecord.getParent(), measRecord.getFootprint())
                              for measRecord in sources}
        noiConfig       =   NoiseReplacerConfig()
        noiseReplacer   =   NoiseReplacer(noiConfig, exposure, footprints)
        #get the badBits
        mask    =   exposure.getMaskedImage().getMask()
        badBits =   self.getBadBits(mask)
        othBits =   mask.getPlaneBitMask('OTHERDET')
        thisBits=   mask.getPlaneBitMask('THISDET')
        measParentCat   =   sources.getChildren(0)
        if self.config.doPlot:
            hduList=pyfits.HDUList()
            #measParentCat   =   measParentCat[80:120] 
        else:
            hduList=None
        for parentIdx, measParentRecord in enumerate(measParentCat):
            # First insert the parent footprint, and measure that
            noiseReplacer.insertSource(measParentRecord.getId())
            self.oneMeasure(measParentRecord,exposure,psfExp,corData,badBits,othBits,thisBits,hduList) 
            noiseReplacer.removeSource(measParentRecord.getId())
            # Then get all the children of this parent, insert footprint in turn, and measure
            measChildCat = sources.getChildren(measParentRecord.getId())
            for ss in measChildCat:
                noiseReplacer.insertSource(ss.getId())
                self.oneMeasure(ss,exposure,psfExp,corData,badBits,othBits,thisBits,hduList) 
                noiseReplacer.removeSource(ss.getId())
        noiseReplacer.end()
        if self.config.doPlot:
            hduList.writeto('debugImgNew.fits' )
        return sources
    
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

