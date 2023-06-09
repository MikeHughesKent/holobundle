# -*- coding: utf-8 -*-
"""
Kent-CAS: Camera Acquisition System

Threading class for image processing of fibre bundle inline holography images. This
inherits from ImageProcessorThread which contains core functionality. 

@author: Mike Hughes
Applied Optics Group
University of Kent

"""

import sys
import logging

import numpy as np
import time

from ImageProcessorThread import ImageProcessorThread
import pybundle
from pybundle import PyBundle
import pyholoscope


import matplotlib.pyplot as plt

class InlineBundleProcessor(ImageProcessorThread):
    
    method = None
    mask = None
    crop = None
    filterSize = None
    calibration = None
    refocus = False
    pyb = None
    preProcessFrame = None
    autoFocusFlag = False
    srCalibrateFlag = False
    invert = False
    showPhase = False
    sr = False
    batchProcessNum = 1
    
    def __init__(self, inBufferSize, outBufferSize, **kwargs):
        
        super().__init__(inBufferSize, outBufferSize, **kwargs)
        self.pyb = PyBundle()
        self.holo = pyholoscope.Holo(pyholoscope.INLINE_MODE, 1, 1)
        
                
    def process_frame(self, inputFrame):
        """ This is called by the thread whenever a frame needs to be processed"""
  
        if self.sr == True:
           # Check we have a list of images, otherwise return None

           if not inputFrame.ndim > 2:
              print("SR but no list of images")
              return None
           #t1 = time.perf_counter()
           #img = inputFrame[0]
           #imgs = np.zeros((np.shape(img)[0], np.shape(img)[1], self.batchProcessNum))
           #imgs[:,:,0] = img
         
           # Pull the rest of images off queue and put in array
           #for idx, img in enumerate(inputFrame):
           #    imgs[:,:,idx] = inputFrame[idx]
           
           #if true:
           #fig, axs = plt.subplots(2, 2)
           ###plt.title("Before Sort")
           #axs[0,0].imshow(imgs[:,:,0])
           ##axs[0,1].imshow(imgs[:,:,1])
          # axs[1,0].imshow(imgs[:,:,2])
          # axs[1,1].imshow(imgs[:,:,3])

           imgs = pybundle.SuperRes.sort_sr_stack(inputFrame, self.batchProcessNum - 1)    
          
           if False:
               fig, axs = plt.subplots(2, 2)
               plt.title("After Sort")
               axs[0,0].imshow(imgs[:,:,0])
               axs[0,1].imshow(imgs[:,:,1])
               axs[1,0].imshow(imgs[:,:,2])
               axs[1,1].imshow(imgs[:,:,3])
           
           #print("num images reconing with ", np.shape(imgs) )
 
           if imgs is not None:
               outputFrame =  self.pyb.process(imgs)   
               self.preProcessFrame = outputFrame
        
        
        else:   # Not Superresolution
           
           # In case we have a list of images instead of an image 
            #if inputFrame.__class__ == list:
            #    inputFrame = inputFrame[0]
            if inputFrame.ndim == 3:
                inputFrame = inputFrame[:,:,0]
            outputFrame =  self.pyb.process(inputFrame)
           
            self.preProcessFrame = outputFrame
    
        if self.refocus == True and outputFrame is not None:
            outputFrame = self.holo.process(outputFrame)
            if self.showPhase is False:
                outputFrame = np.abs(outputFrame)
                if self.invert is True:
                    outputFrame = np.max(outputFrame) - outputFrame
            else:
                 outputFrame = np.angle(outputFrame)    
            if outputFrame is not None:
                outputFrame = np.abs(outputFrame)   # Take intensity from complex image
                
            
        return outputFrame



    def handle_flags(self):
        """ Flags can be set externally for actions which cannot be performed
        until one or more images are available. Flags are checked every time we process a new image 
        """
       
        # AUTOFOCUS
        if self.autoFocusFlag:
             self.autoFocusFlag = False
             t1 = time.perf_counter()
            
            
        # SUPER RESOLUTION CALIBRATION    
        if self.srCalibrateFlag:
            if self.get_num_images_in_input_queue() >= self.batchProcessNum:
                self.calibrate_sr()
                # Remove flag
                self.srCalibrateFlag = False
                    
    def calibrate_sr(self):
        
        #if len(self.currentInputImage) >= self.batchProcessNum:
        
              # Convert list of images to 3D numpy array
              #img = self.currentInputImage[0]
              #imgs = np.zeros((np.shape(img)[0], np.shape(img)[1], self.batchProcessNum))
              #imgs[:,:,0] = img
            
              #for idx, img in enumerate(self.currentInputImage):
              #    imgs[:,:,idx] = self.currentInputImage[idx]
              
              # Extract a sequence of frames in correct order following blank reference frame
              calibImgs = pybundle.SuperRes.sort_sr_stack(self.currentInputImage, self.batchProcessNum - 1)    
        
              # SR Calibration
              self.pyb.set_sr_calib_images(calibImgs)
              self.pyb.calibrate_sr()
    
    #def acquire_sr_backgrounds(self):
    #    backImgs = pybundle.SuperRes.sort_sr_stack(self.currentInputImage, self.batchProcessNum - 1)    
    
    #    self.pyb.set_sr_backgrounds(backImgs)
                    
                    
    def auto_focus(self, **kwargs):
        
        if self.preProcessFrame is not None:
            return self.holo.auto_focus(self.preProcessFrame.astype('float32'), **kwargs)
        
        
    def update_settings(self):
        """ For compatibility with multi-processor version"""
        pass