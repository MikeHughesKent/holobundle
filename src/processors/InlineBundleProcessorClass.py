# -*- coding: utf-8 -*-
"""
Kent-CAS: Camera Acquisition System

Class for image processing of fibre bundle inline holography images. This
inherits from ImageProcessorThread which contains core functionality. 

@author: Mike Hughes
Applied Optics Group
University of Kent

"""

import sys
import logging

import numpy as np
import time

from ImageProcessorClass import ImageProcessorClass

import pybundle
from pybundle import PyBundle
import pyholoscope


import matplotlib.pyplot as plt

class InlineBundleProcessorClass(ImageProcessorClass):
    
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
    differential = False
    
    def __init__(self, **kwargs):
        
        super().__init__()
        self.pyb = PyBundle()
        self.holo = pyholoscope.Holo(pyholoscope.INLINE_MODE, 1, 1)
        
                
    def process(self, inputFrame):
        """ This is called by the thread whenever a frame needs to be processed"""
        outputFrame = None
        if self.sr == True:
           # Check we have a list of images, otherwise return None

           if not inputFrame.ndim > 2:
              print("SR but no list of images")
              return None
           
            
            
           # fig, axs = plt.subplots(2, 4, dpi=150)
           # fig.suptitle('Raw', fontsize=16)
           # axs[0,0].imshow(inputFrame[:,:,0])
           # axs[0,1].imshow(inputFrame[:,:,1])
           # axs[0,2].imshow(inputFrame[:,:,2])
           # axs[0,3].imshow(inputFrame[:,:,3])
           # axs[1,0].imshow(inputFrame[:,:,4])
           # axs[1,1].imshow(inputFrame[:,:,5])
           # axs[1,2].imshow(inputFrame[:,:,6])
           # axs[1,3].imshow(inputFrame[:,:,7])


           imgs = pybundle.SuperRes.sort_sr_stack(inputFrame, self.batchProcessNum - 1)  
           
           # fig, axs = plt.subplots(2, 4, dpi=150)
           # fig.suptitle('Sorted', fontsize=16)
           # axs[0,0].imshow(imgs[:,:,0])
           # axs[0,1].imshow(imgs[:,:,1])
           # axs[0,2].imshow(imgs[:,:,2])
           # axs[0,3].imshow(imgs[:,:,3])
           # axs[1,0].imshow(imgs[:,:,4])
           # axs[1,1].imshow(imgs[:,:,5])
           # axs[1,2].imshow(imgs[:,:,6])


          
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
        
        
        elif self.differential:   # Differential Mode
            print(inputFrame.ndim)
            if inputFrame.ndim == 3:
                if np.shape(inputFrame)[2] == 2:
                    print("proc diff")
                    outputFrame = inputFrame[:,:,0] - inputFrame[:,:,1]
                    outputFrame = self.pyb.process(outputFrame)
                    self.preProcessFrame = outputFrame


        else:   # Standard Mode
            # In case we have a list of images instead of an image 
            #if inputFrame.__class__ == list:
            #    inputFrame = inputFrame[0]
            if inputFrame.ndim == 3:
                inputFrame = inputFrame[:,:,0]
            outputFrame = self.pyb.process(inputFrame)
           
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
                
                
    def set_differential(self, isDifferential):
        self.differential = isDifferential
            
                    
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
              
   
    def capture_sr_shift(self):
        return pybundle.SuperRes.sort_sr_stack(self.currentInputImage, self.batchProcessNum - 1)    
        
        
        
        
        
    #def acquire_sr_backgrounds(self):
    #    backImgs = pybundle.SuperRes.sort_sr_stack(self.currentInputImage, self.batchProcessNum - 1)    
    
    #    self.pyb.set_sr_backgrounds(backImgs)
                    
                    
    def auto_focus(self, **kwargs):
        
        if self.preProcessFrame is not None:
            return self.holo.auto_focus(self.preProcessFrame.astype('float32'), **kwargs)
        
        
    def update_settings(self):
        """ For compatibility with multi-processor version"""
        pass