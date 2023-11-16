# -*- coding: utf-8 -*-
"""
HoloBundle
Inline Fibre Bundle Holography GUI

A QT based graphical user interface for inline holographic microscopy through
fibre imaging bundles developed in the Applied Optics Group, School of Physics
and Astronomy, University of Kent. 

The GUI can interface with several different camera models or process saved files.
 
The PyFibreBundle package is used for pre-processing fibre bundle images and 
the PyHoloscope package is used for holography. The GUI is based on the CAS-GUI
developed at University of Kent.

This class inherits from CAS_GUI_Bundle and CAS_GUI_Base which provides
most of the functionality for acquiring images and removing the core pattern.
This file handles only the holography specific elements of processing as well
as designing the layout of the GUI and creating the holography processing panel
and the refocusing panel. Super-resolution GUI elements are also implemented
here.

@author: Mike Hughes, Applied Optics Group, Physics and Astronomy, University of Kent

"""

import sys 
import os
from pathlib import Path


import serial


file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, 'processors')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../../cas/src')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../../cas/src/widgets')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../../cas/src/cameras')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../../cas/src/threads')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../../pyfibrebundle/src')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../../pyholoscope/src')))


import time
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette, QColor, QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5.QtGui import QPainter, QBrush, QPen

from PIL import Image
import cv2 as cv

from CAS_GUI_Base import CAS_GUI
from CAS_GUI_Bundle import CAS_GUI_Bundle

from image_display import ImageDisplay

from cam_control_panel import *
from ImageAcquisitionThread import ImageAcquisitionThread
from InlineBundleProcessor import InlineBundleProcessor

import pyholoscope

import pybundle
from pybundle import PyBundle
from pybundle import SuperRes


# Led Modes
SEQUENTIAL = 0
SINGLE = 1

class Holo_Bundle(CAS_GUI_Bundle):
    
    authorName = "AOG"
    appName = "HoloBundle"
    windowTitle = "HoloBundle"
    logoFilename = "../res/kent_logo_2.png"
    
    cuda = True
    srBackgrounds = None   
    sr = True
    mosaicingEnabled = False
    sr_single_led_id = 1  
    
    def __init__(self,parent=None):        
        
        
        # If we are doing Super Res try to open the serial comms to the LED driver
        if self.sr:
             try: 
                 self.serial = serial.Serial('COM3', 9600, timeout=0,
                      parity=serial.PARITY_EVEN, rtscts=1)
                 self.serial.reset_output_buffer()
                 time.sleep(1)    # Otherwise it seems not to work, not sure why
                 
             except:
                 print("cannot open serial")
                 self.serial = None
          
        
        # Simulated camera used this file for images
        self.sourceFilename = r"C:\Users\AOG\Dropbox\Programming\Python\cas\tests\test_data\stack_10.tif"
        
        self.sourceFilename = r"..\tests\test_data\sr_test_1_background.tif"
        self.sourceFilename = r"..\tests\test_data\sr_test_1.tif"

        
        self.controlPanelSize = 220
        self.rawImageBufferSize = 20

        super(Holo_Bundle, self).__init__(parent)        
        
        # Call these functions to update things based on the default GUI options
        self.handle_change_show_bundle_control(1)
        self.handle_sr_clear_shifts()
        self.handle_sr_enabled()
        
        # Set a black background on the window
        self.set_colour_scheme('black')


        self.exportStackDialog = ExportStackDialog()

        #self.load_background()
        #self.handle_load_calibration()
       # self.handle_load_sr_background()

        
    
    def create_layout(self):
        """ Called by parent class to assemble the GUI from Qt Widgets"""
        
        
        super().create_layout()
        self.setWindowIcon(QtGui.QIcon('../res/icon.png'))
        
        # Add custom buttons to main menu
        self.mainMenuLoadBackBtn = QPushButton('Load Background File')
        self.mainMenuLayout.addWidget(self.mainMenuLoadBackBtn)
        self.mainMenuLoadBackBtn.clicked.connect(self.load_background_from_click)
        
        self.mainMenuCalibrateBtn = QPushButton('Calibrate')
        self.mainMenuLayout.addWidget(self.mainMenuCalibrateBtn)
        self.mainMenuCalibrateBtn.clicked.connect(self.handle_calibrate)
        
        # Create the processing panels
        self.holoPanel = self.init_inline_holo_process_panel(self.controlPanelSize)
        self.refocusPanel = self.init_refocus_panel(self.controlPanelSize)
        self.srPanel = self.init_inline_holo_sr_panel(self.controlPanelSize)
        
        # Create panel with 'Show processing options' checkbox
        self.visibilityControl = QWidget()
        self.visibilityControl.setLayout(visLayout:= QVBoxLayout())
        self.showBundleControlCheck = QCheckBox("Show Processing Options", objectName = "showBundleControlCheck")
        self.showBundleControlCheck.toggled.connect(self.handle_change_show_bundle_control)
        visLayout.addWidget(self.showBundleControlCheck)
        visLayout.addStretch()
        
        self.layout.addWidget(self.holoPanel)
        self.layout.addWidget(self.srPanel)


        self.topLayout.insertWidget(self.topLayout.count() - 1,self.refocusPanel)
        self.topLayout.insertWidget(self.topLayout.count() - 1,self.visibilityControl)
      
    
        
        
        
    def init_refocus_panel(self, panelSize):
        """Create the panel with the depth slider"""
        
        
        groupBox = QGroupBox("Refocusing")
        groupBox.setLayout(layout:=QVBoxLayout())
       
        self.holoDepthSlider = QSlider(QtCore.Qt.Horizontal, objectName = 'holoDepthSlider')
        self.holoDepthSlider.setTickPosition(QSlider.TicksBelow)
        self.holoDepthSlider.setTickInterval(10)
        self.holoDepthSlider.setMaximum(5000)
        self.holoDepthSlider.valueChanged[int].connect(self.handle_depth_slider)       
        
        self.holoDepthInput = QDoubleSpinBox(objectName='holoDepthInput')
        self.holoDepthInput.setKeyboardTracking(False)
        self.holoDepthInput.setMaximum(10**6)
        self.holoDepthInput.setMinimum(-10**6)
        self.holoDepthInput.setSingleStep(10)
       
        self.holoAutoFocusBtn=QPushButton('Auto Depth')
        
        layout.addWidget(QLabel('Refocus depth (microns):'))
        layout.addWidget(self.holoDepthSlider)
        layout.addWidget(self.holoDepthInput)        
        layout.addWidget(self.holoAutoFocusBtn)

        self.holoAutoFocusBtn.clicked.connect(self.auto_focus_click)
        self.holoDepthInput.valueChanged[float].connect(self.handle_changed_bundle_processing)

        # Export Stack
        self.holoCreateStackBtn = QPushButton('Export Depth Stack')
        self.holoCreateStackBtn.clicked.connect(self.create_depth_stack_click)
        layout.addWidget(self.holoCreateStackBtn)

        
        return groupBox   
         

    def init_inline_holo_process_panel(self, panelSize):
        """ Create the panel with full controls for holography"""        
        
        self.holoWavelengthInput = QDoubleSpinBox(objectName='holoWavelengthInput')
        self.holoWavelengthInput.setMaximum(10**6)
        self.holoWavelengthInput.setMinimum(-10**6)
        self.holoWavelengthInput.setDecimals(3)
        
        self.holoPixelSizeInput = QDoubleSpinBox(objectName='holoPixelSizeInput')
        self.holoPixelSizeInput.setMaximum(10**6)
        self.holoPixelSizeInput.setMinimum(-10**6)
        
            
        self.holoRefocusCheck = QCheckBox("Refocus", objectName='holoRefocusCheck')
        self.holoPhaseCheck = QCheckBox("Show Phase", objectName='holoPhaseCheck')
        self.holoInvertCheck = QCheckBox("Invert Image", objectName='holoInvertCheck')
        
        self.holoWindowCombo = QComboBox(objectName='holoWindowCombo')
        self.holoWindowCombo.addItems(['None', 'Circular', 'Rectangular'])

        self.holoWindowThicknessInput = QDoubleSpinBox(objectName='holoWindowThicknessInput')
        self.holoWindowThicknessInput.setMaximum(10**6)
        self.holoWindowThicknessInput.setMinimum(-10**6)
        
        self.holoAutoFocusMinInput = QDoubleSpinBox(objectName='holoAutoFocusMinInput')
        self.holoAutoFocusMinInput.setMaximum(10**6)
        self.holoAutoFocusMinInput.setMinimum(-10**6)
        
        self.holoAutoFocusMaxInput = QDoubleSpinBox(objectName='holoAutoFocusManInput')
        self.holoAutoFocusMaxInput.setMaximum(10**6)
        self.holoAutoFocusMaxInput.setMinimum(-10**6)
        
        
        self.holoAutoFocusCoarseDivisionsInput = QDoubleSpinBox(objectName='holoAutoFocusCoarseDivisionsInput')
        self.holoAutoFocusCoarseDivisionsInput.setMaximum(10**6)
        self.holoAutoFocusCoarseDivisionsInput.setMinimum(0)
        
        self.holoAutoFocusROIMarginInput = QDoubleSpinBox(objectName='holoAutoFocusROIMarginInput')
        self.holoAutoFocusROIMarginInput.setMaximum(10**6)
        self.holoAutoFocusROIMarginInput.setMinimum(0)
        
        self.holoSliderMaxInput = QSpinBox(objectName='holoSliderMaxInput')
        self.holoSliderMaxInput.setMaximum(10**6)
        self.holoSliderMaxInput.setMinimum(0)
        self.holoSliderMaxInput.setKeyboardTracking(False)
        
        holoPanel = QWidget()
        holoPanel.setLayout(topLayout:=QVBoxLayout())
        holoPanel.setMaximumWidth(panelSize)
        holoPanel.setMinimumWidth(panelSize)        
        
        groupBox = QGroupBox("Inline Holography")
        topLayout.addWidget(groupBox)
        groupBox.setLayout(layout:=QVBoxLayout())
        
        layout.addWidget(QLabel('Wavelegnth (microns):'))
        layout.addWidget(self.holoWavelengthInput)
        layout.addWidget(QLabel('Pixel Size (microns):'))
        layout.addWidget(self.holoPixelSizeInput)       
        
        layout.addWidget(self.holoRefocusCheck)
        layout.addWidget(self.holoPhaseCheck)
        layout.addWidget(self.holoInvertCheck)
         
        layout.addWidget(QLabel('Window:'))
        layout.addWidget(self.holoWindowCombo)

        layout.addWidget(QLabel("Window Thickness (px):"))
        layout.addWidget(self.holoWindowThicknessInput) 
        
        layout.addWidget(QLabel("Autofocus Min (microns):"))
        layout.addWidget(self.holoAutoFocusMinInput) 
        
        layout.addWidget(QLabel("Autofocus Max (microns):"))
        layout.addWidget(self.holoAutoFocusMaxInput)
        
        layout.addWidget(QLabel("Autofocus Coarse Intervals:"))
        layout.addWidget(self.holoAutoFocusCoarseDivisionsInput)
        
        layout.addWidget(QLabel("Autofocus ROI Margin (px):"))
        layout.addWidget(self.holoAutoFocusROIMarginInput)
        
        layout.addWidget(QLabel("Depth Slider Max (microns):"))
        layout.addWidget(self.holoSliderMaxInput)

        topLayout.addStretch()
        
        # We call handle_change_bundle_processing because we have overloaded this from CAS_GUI_Bundle and
        # its convenient to have one functions that handles all updates to processing
        self.holoWavelengthInput.valueChanged[float].connect(self.handle_changed_bundle_processing)
        self.holoPixelSizeInput.valueChanged[float].connect(self.handle_changed_bundle_processing)
        self.holoRefocusCheck.stateChanged.connect(self.handle_changed_bundle_processing)
        self.holoPhaseCheck.stateChanged.connect(self.handle_changed_bundle_processing)
        self.holoInvertCheck.stateChanged.connect(self.handle_changed_bundle_processing)
        self.holoWindowThicknessInput.valueChanged[float].connect(self.handle_changed_bundle_processing)
        self.holoWindowCombo.currentIndexChanged[int].connect(self.handle_changed_bundle_processing)
        self.holoSliderMaxInput.valueChanged[int].connect(self.handle_changed_bundle_processing)

        return holoPanel    


    def init_inline_holo_sr_panel(self, panelSize):
        """Create the panel for super-resolution"""
        
        srPanel = QWidget()
        srPanel.setLayout(topLayout:=QVBoxLayout())
        srPanel.setMaximumWidth(panelSize)
        srPanel.setMinimumWidth(panelSize)  
        
        groupBox = QGroupBox("Super Resolution")
        groupBox.setLayout(layout:=QVBoxLayout())    
        
        self.srCalibBtn=QPushButton('Calibrate Super Resolution')
        self.srEnabledCheck = QCheckBox('Enable Super Resolution', objectName = 'srEnabledCheck')
        
        self.srSaveCalibBtn=QPushButton('Save SR Calibration')
        self.srLoadCalibBtn=QPushButton('Load SR Calibration')
        
        self.srAcquireBackgroundsBtn=QPushButton('Acquire SR Background')
        self.srSaveBackgroundsBtn=QPushButton('Save SR Background')
        self.srLoadBackgroundsBtn=QPushButton('Load SR Background') 
        
        
        self.srNumShiftsInput = QSpinBox(objectName='srNumShiftsInput')
        self.srNumShiftsInput.setMaximum(10**6)
        self.srNumShiftsInput.setMinimum(0)
        self.srNumShiftsInput.setKeyboardTracking(False)
        self.srMultiBackgroundsCheck = QCheckBox('Use Background Stack', objectName = 'srMultiBackgrounds')
        self.srMultiNormalisationCheck = QCheckBox('Use Normalisation Stack', objectName = 'srMultiNormalisation')
        self.srCaptureShiftBtn = QPushButton('Capture Shift')
        self.srGenerateLUTBtn = QPushButton('Generate Calibration LUT')
        self.srUseLUTCheck = QCheckBox('Use Calibration LUT')
        self.srLUTMinInput = QDoubleSpinBox(objectName = 'srLUTMinInput')
        self.srLUTMinInput.setMaximum(10000)
        self.srLUTMaxInput = QDoubleSpinBox(objectName = 'srLUTMaxInput')
        self.srLUTMaxInput.setMaximum(10000)
        self.srLUTNumStepsInput = QSpinBox(objectName = 'srLUTNumStepsInput')
        self.srLUTMinInput.setMaximum(1000)

        self.srSaveCalibrationLUTBtn=QPushButton('Save Calibration LUT')
        self.srLoadCalibrationLUTBtn=QPushButton('Load Calibration LUT') 


        
        layout.addWidget(self.srCalibBtn)
        layout.addWidget(self.srEnabledCheck)
        layout.addWidget(QLabel("Number of shifts:"))
        layout.addWidget(self.srNumShiftsInput)
        layout.addWidget(self.srSaveCalibBtn)
        layout.addWidget(self.srLoadCalibBtn)
        layout.addWidget(self.srAcquireBackgroundsBtn)
        layout.addWidget(self.srSaveBackgroundsBtn)
        layout.addWidget(self.srLoadBackgroundsBtn)
        layout.addWidget(self.srMultiBackgroundsCheck)
        layout.addWidget(self.srMultiNormalisationCheck)
        layout.addWidget(self.srUseLUTCheck)
        layout.addWidget(self.srCaptureShiftBtn)

        layout.addWidget(self.srGenerateLUTBtn)
        
        layout.addWidget(QLabel('LUT Min Depth (microns):'))
        layout.addWidget(self.srLUTMinInput)
        layout.addWidget(QLabel('LUT Max Depth (microns):'))
        layout.addWidget(self.srLUTMaxInput)
        layout.addWidget(QLabel('LUT Num Steps:'))
        layout.addWidget(self.srLUTNumStepsInput)
        layout.addWidget(self.srSaveCalibrationLUTBtn)
        layout.addWidget(self.srLoadCalibrationLUTBtn)
        
        
        
        self.plotButton = QPushButton("Plot")
        self.plotButton.clicked.connect(self.handle_plot_button)
        #layout.addWidget(self.plotButton)
        
        topLayout.addWidget(groupBox)
        topLayout.addStretch()
      
        self.srCalibBtn.clicked.connect(self.handle_sr_calibrate_click)
        self.srEnabledCheck.stateChanged.connect(self.handle_sr_enabled)
        self.srSaveCalibBtn.clicked.connect(self.handle_save_sr_calib)
        self.srLoadCalibBtn.clicked.connect(self.handle_load_sr_calib)
        self.srAcquireBackgroundsBtn.clicked.connect(self.handle_acquire_sr_background)
        self.srSaveBackgroundsBtn.clicked.connect(self.handle_save_sr_background)
        self.srLoadBackgroundsBtn.clicked.connect(self.handle_load_sr_background)
        self.srMultiBackgroundsCheck.stateChanged.connect(self.handle_changed_bundle_processing)
        self.srMultiNormalisationCheck.stateChanged.connect(self.handle_changed_bundle_processing)
        self.srUseLUTCheck.stateChanged.connect(self.handle_changed_bundle_processing)
        
        self.holoWindowThicknessInput.valueChanged[float].connect(self.handle_changed_bundle_processing)
        self.srGenerateLUTBtn.clicked.connect(self.handle_sr_generate_LUT)
        self.srCaptureShiftBtn.clicked.connect(self.handle_sr_capture_shift)
        
        self.srSaveCalibrationLUTBtn.clicked.connect(self.handle_sr_save_calibration_lut)
        self.srLoadCalibrationLUTBtn.clicked.connect(self.handle_sr_load_calibration_lut)

        
        
        return srPanel   
    
    
    
    
    def handle_plot_button(self, event):
        fig, axs = plt.subplots(2, 4, dpi=150)
        axs[0,0].imshow(self.imageProcessor.currentInputImage[:,:,0])
        axs[0,1].imshow(self.imageProcessor.currentInputImage[:,:,1])
        axs[0,2].imshow(self.imageProcessor.currentInputImage[:,:,2])
        axs[0,3].imshow(self.imageProcessor.currentInputImage[:,:,3])
        axs[1,0].imshow(self.imageProcessor.currentInputImage[:,:,4])
        axs[1,1].imshow(self.imageProcessor.currentInputImage[:,:,5])
        axs[1,2].imshow(self.imageProcessor.currentInputImage[:,:,6])



        
    def create_processors(self):
        """ Create image processor thread"""    
        if self.imageThread is not None:
            inputQueue = self.imageThread.get_image_queue()
        else:
            inputQueue = None   # This will force Processor to create its own input queue
        
        if self.imageProcessor is None:
            self.imageProcessor = InlineBundleProcessor(10,10, inputQueue = inputQueue, acquisitionLock = self.acquisitionLock)
            if self.imageProcessor is not None:
                self.imageProcessor.start()
        
        self.handle_changed_bundle_processing()
        
        
    

    def update_image_display(self):
        """ Puts either raw or (if available) processed image on display"""       
        if self.bundleShowRaw.isChecked():
           if self.currentImage is not None:
               self.mainDisplay.set_mono_image(self.currentImage)
        else:
           if self.currentProcessedImage is not None:
               self.mainDisplay.set_mono_image(self.currentProcessedImage)
           else:
               if self.currentImage is not None:
                   self.mainDisplay.set_mono_image(self.currentImage)
                   
                   
                   
                   
                   
            
    
    def handle_changed_bundle_processing(self):   
        """Called when changes are made to the options pane. Updates the processor thread."""
        
        # Check the slider matches the input box value and the max value is correct
        self.holoDepthSlider.setValue(int(self.holoDepthInput.value()))
        self.holoDepthSlider.setMaximum(int(self.holoSliderMaxInput.value()))

        # Holography specific processing
        if self.imageProcessor is not None:
            self.imageProcessor.showPhase = self.holoPhaseCheck.isChecked()
            self.imageProcessor.invert = self.holoInvertCheck.isChecked()
            self.imageProcessor.holo.cuda = self.cuda   
            if self.holoRefocusCheck.isChecked():
                
                self.imageProcessor.refocus = True
                
                # Wavelength
                if self.holoWavelengthInput.value() != self.imageProcessor.holo.wavelength / 10**6:
                    self.imageProcessor.holo.set_wavelength(self.holoWavelengthInput.value()/ 10**6)
                
                # Depth
                if self.holoDepthInput.value() != self.imageProcessor.holo.depth / 10**6:
                    self.imageProcessor.holo.set_depth(self.holoDepthInput.value()/ 10**6)
                
                # Windowing
                if self.holoWindowCombo.currentText() == "Circular":
                    self.imageProcessor.holo.set_auto_window(True)
                    self.imageProcessor.holo.set_window_shape('circle')                    
                    self.imageProcessor.holo.set_window_radius(None)
                    self.imageProcessor.holo.set_window_thickness(self.holoWindowThicknessInput.value())
                else:
                    self.imageProcessor.holo.window = None
            else:
                self.imageProcessor.refocus = False
        
        # The basic bundle processing is defined in CAS_GUI_Bundle
        super().handle_changed_bundle_processing()
       
        # Have to do pixel size last in case bundle processing changed as this
        # may change scale factor. If we changed the target pixel size we then also
        # have to call update_file_processing again, which was called by 
        # handle_changed_bundle_processing in the parent class _before_ we
        # updated the pixel size
        
        if self.imageProcessor is not None:
            scaleFactor = self.imageProcessor.pyb.get_pixel_scale() 
            if scaleFactor is not None:
                targetPixelSize = self.imageProcessor.pyb.get_pixel_scale() * self.holoPixelSizeInput.value() / 10**6
            else:
                targetPixelSize = self.holoPixelSizeInput.value() / 10**6

            if targetPixelSize != self.imageProcessor.holo.pixelSize:
                self.imageProcessor.holo.set_pixel_size(targetPixelSize)
                self.update_file_processing()

            if self.srEnabledCheck.isChecked():
                
                self.imageProcessor.sr = True
                self.imageProcessor.pyb.set_super_res(True)
                self.imageProcessor.pyb.set_sr_backgrounds(self.srBackgrounds)
                self.imageProcessor.pyb.set_sr_normalisation_images(self.srBackgrounds)
                self.imageProcessor.pyb.set_sr_multi_normalisation(self.srMultiNormalisationCheck.isChecked())
                self.imageProcessor.pyb.set_sr_multi_backgrounds(self.srMultiBackgroundsCheck.isChecked())
                self.imageProcessor.set_batch_process_num(self.srNumShiftsInput.value() + 1)
                self.imageProcessor.pyb.set_sr_use_lut(self.srUseLUTCheck.isChecked())
                self.imageProcessor.pyb.set_sr_param_value(self.holoDepthInput.value()/ 10**6)

                if self.imageThread is not None:
                    self.imageThread.set_num_removal_when_full(self.srNumShiftsInput.value() + 1)
                self.update_file_processing()
            else:
                self.imageProcessor.sr = False
                self.imageProcessor.pyb.set_super_res(False)
                self.imageProcessor.set_batch_process_num(1)
                self.update_file_processing()


    def handle_sr_enabled(self):        
          
          if self.serial is not None:
              if self.srEnabledCheck.isChecked():
                  self.sr_set_led_mode(SEQUENTIAL)
              else:    
                  self.sr_set_led_mode(SINGLE)
          if self.cam is not None:
              if self.srEnabledCheck.isChecked():
                  self.cam.set_trigger_mode(True)
              else:
                  self.cam.set_trigger_mode(False)

          self.update_camera_ranges()
          self.handle_changed_bundle_processing()    
      
      
    def sr_set_led_mode(self, mode):
          """ If using an array of LEDs, communicated with Arduino to set correct operation 
          for current operation mode.
          
          """
          if self.serial is not None:
              self.serial.reset_output_buffer()
      
              if mode == SEQUENTIAL:
                  print("writing multi")
                  self.serial.write(b'm')
              if mode == SINGLE:
                  print("writing single")
                  self.serial.write(b's' + str(self.sr_single_led_id) + '\n')   
                  
                  
    def handle_change_show_bundle_control(self, event):
        """Called when checkbox to show processing controls is toggled"""

        if self.showBundleControlCheck.isChecked():
            self.bundleProcessPanel.show()
            self.holoPanel.show()
            self.srPanel.show()
        else:  
            self.bundleProcessPanel.hide() 
            self.holoPanel.hide()
            self.srPanel.hide()

    
    
    def auto_focus_click(self):
        """ Finds best focus and update depth slider"""
     
        if self.mainDisplay.roi is not None:
            roi = PyHoloscope.roi(self.mainDisplay.roi[0], self.mainDisplay.roi[1], self.mainDisplay.roi[2] - self.mainDisplay.roi[0], self.mainDisplay.roi[3] - self.mainDisplay.roi[1])
        else:
            roi = None
        autofocusMax = self.holoAutoFocusMaxInput.value() / 1000
        autofocusMin = self.holoAutoFocusMinInput.value() / 1000
        numSearchDivisions = int(self.holoAutoFocusCoarseDivisionsInput.value())
        autofocusROIMargin = self.holoAutoFocusROIMarginInput.value()
        if self.imageThread is not None:
            self.imageThread.pause()
        autoFocus = (self.imageProcessor.auto_focus(roi = roi, margin = autofocusROIMargin, depthRange = (autofocusMin, autofocusMax), coarseSearchInterval = numSearchDivisions))
        self.holoDepthInput.setValue(autoFocus * 1000) 

        if self.imageThread is not None:
            self.imageThread.resume()


    def handle_depth_slider(self):
        """ Called when depth slider is changed. This will change the input box
        which will then trigger handle_changed_bundle_processing where the processing will
        be updated
        """
        self.holoDepthInput.setValue(int(self.holoDepthSlider.value()))
        
        
    def handle_sr_generate_LUT(self):
        """ Called when SR Generate LUT button is clicked.
        """
        param_depths = np.array(self.sr_param_depths)
        param_holograms = np.stack(self.sr_param_holograms, axis = 3)
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
       
        if self.imageThread is not None: self.imageThread.pause() 
        if self.imageProcessor is not None: self.imageProcessor.pause()
               
        #t1 = time.perf_counter()
        self.srParamShiftCalib = pybundle.SuperRes.calib_param_shift(param_depths, param_holograms, self.imageProcessor.pyb.calibration, forceZero = True)
        self.imageProcessor.pyb.set_calib_image(self.backgroundImage)
        self.imageProcessor.pyb.calibrate_sr_lut(self.srParamShiftCalib, (self.srLUTMinInput.value() / 10**6, self.srLUTMaxInput.value() / 10**6), self.srLUTNumStepsInput.value())
        #print(f"LUT took {time.perf_counter() -t1} to build.")
        
        if self.imageThread is not None: self.imageThread.resume()
        if self.imageProcessor is not None: self.imageProcessor.resume()
        QApplication.restoreOverrideCursor()
        
        
        
        
    def handle_sr_capture_shift(self):
        """
        """
        if self.imageProcessor.pyb.calibration is None:
            QMessageBox.about(self, "Error", "Shift measurement requires an interpolation calibration.")  
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.sr_param_holograms.append(self.imageProcessor.capture_sr_shift())
        self.sr_param_depths.append(self.imageProcessor.holo.depth)
        QApplication.restoreOverrideCursor()

    
   
    def handle_sr_clear_shifts(self):
        """
        """
        self.sr_param_holograms = []
        self.sr_param_depths = []



    def handle_sr_save_calibration_lut(self):  
        with open('sr_calib_lut.dat','wb') as pickleFile:
            pickle.dump(self.imageProcessor.pyb.srCalibrationLUT, pickleFile)

    
        
        
    def handle_sr_load_calibration_lut(self):  
        with open('sr_calib_lut.dat', 'rb') as pickleFile:
            self.imageProcessor.pyb.srCalibrationLUT = pickle.load(pickleFile)
        self.handle_changed_bundle_processing()  
        print(self.imageProcessor.pyb.srCalibrationLUT)
        
        
        
    def handle_sr_calibrate_click(self): 
        """ Called when SR Calibrate is clicked, flage the imageProcessor to
        calibrate once images are available
        """
        if not self.srMultiBackgroundsCheck.isChecked() and self.backgroundImage is None:
            QMessageBox.about(self, "Error", "Background file required.")  
            return    
      
        if self.srMultiBackgroundsCheck.isChecked() and self.srBackgrounds is None:
            QMessageBox.about(self, "Error", "Background stack required.")  
            return  
      
        if self.imageProcessor is not None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.imageProcessor.pyb.set_calib_image(self.backgroundImage)
            self.imageProcessor.calibrate_sr()
            
            # We also do a conventional calibration at the same time
            self.imageProcessor.pyb.calibrate()            
            QApplication.restoreOverrideCursor()
        
        self.handle_changed_bundle_processing()    
            
    def handle_save_sr_calib(self):
        """ Saves SR calibration to a file"""
        with open('sr_calib.dat','wb') as pickleFile:
            pickle.dump(self.imageProcessor.pyb.calibrationSR, pickleFile)

        
    def handle_load_sr_calib(self):
        """ Loads SR calibration from a file"""
        with open('sr_calib.dat', 'rb') as pickleFile:
            self.imageProcessor.pyb.calibrationSR = pickle.load(pickleFile)
        self.handle_changed_bundle_processing()        
        
        
    def handle_save_sr_background(self):
        """ If we have a stack of backgrounds for SR, save to a TIF stack"""
        if self.srBackgrounds is not None:
            imlist = []

            for idx in range(np.shape(self.srBackgrounds)[2]):
                imlist.append(Image.fromarray(self.srBackgrounds[:,:,idx].astype('uint16')))

            imlist[0].save('sr_backgrounds.tif', compression="tiff_deflate", save_all=True,
                   append_images=imlist[1:])                   

        
    def handle_load_sr_background(self):
        """ Loads a tif stack of images and sets as current super-resolution backgrounds stack."""
        self.dataset = Image.open('sr_backgrounds.tif')
        h = np.shape(self.dataset)[0]
        w = np.shape(self.dataset)[1]
        
        
        imageBuffer = np.zeros((h,w,self.dataset.n_frames))
        
        for i in range(self.dataset.n_frames):
            self.dataset.seek(i)
            imageBuffer[:,:,i] = np.array(self.dataset).astype('double')
        self.dataset.close() 
        
        self.srBackgrounds = imageBuffer
        self.backgroundImage = self.srBackgrounds[:,:,self.sr_single_led_id]
        self.handle_changed_bundle_processing()
        
        
    def handle_acquire_sr_background(self):
        """ Sets the current SR image stack as the background stack and updates processor
        """
        if self.imageProcessor is not None and self.srEnabledCheck.isChecked():
            self.srBackgrounds = pybundle.SuperRes.sort_sr_stack(self.imageProcessor.currentInputImage, self.imageProcessor.batchProcessNum - 1)    
            self.backgroundImage = self.srBackgrounds[:,:,self.sr_single_led_id]
            self.handle_changed_bundle_processing()
            
            
    def apply_default_settings(self):
        """Applies hard-coded default settings. These will then be saved on program
        exit. """
        
        self.showBundleControlCheck.setChecked(True)
        self.camSourceCombo.setCurrentIndex(0)
        self.bundleCentreXInput.setValue(500)
        self.bundleCentreYInput.setValue(500)
        self.bundleRadiusInput.setValue(450)
        self.bundleShowRaw.setChecked(False)
        self.bundleCoreMethodCombo.setCurrentIndex(1)
        self.bundleSubtractBackCheck.setChecked(True)
        self.bundleNormaliseCheck.setChecked(True)
        self.bundleFilterSizeInput.setValue(2)
        self.bundleCropCheck.setChecked(True)
        self.bundleMaskCheck.setChecked(True)
        self.bundleGridSizeInput.setValue(512)
        self.holoWavelengthInput.setValue(0.455)
        self.holoPixelSizeInput.setValue(0.64)
        self.holoRefocusCheck.setChecked(True)
        self.holoPhaseCheck.setChecked(False)
        self.holoInvertCheck.setChecked(False)
        self.holoWindowCombo.setCurrentIndex(1)
        self.holoWindowThicknessInput.setValue(20)
        self.holoAutoFocusMinInput.setValue(100)
        self.holoAutoFocusMaxInput.setValue(1500)
        self.holoAutoFocusCoarseDivisionsInput.setValue(20)
        self.holoAutoFocusROIMarginInput.setValue(20)
        self.holoSliderMaxInput.setValue(3000)
        
        
    def update_file_processing(self):
        """Override. In CAS GUI, update_file_processing calls the processor directly
        which we might not want to if we depend on multiple frames for super-resolution"""
        if not self.srEnabledCheck.isChecked():
            super().update_file_processing()
            
            
    def create_depth_stack_click(self):
        """ Creates a depth stack over a specified range.
        """
        
        if self.imageProcessor is not None and self.imageProcessor.preProcessFrame is not None:
            if self.exportStackDialog.exec():
                try:
                    filename = QFileDialog.getSaveFileName(self, 'Select filename to save to:', '', filter='*.tif')[0]
                except:
                    filename = None
                if filename is not None and filename != '':
                     depthRange = (self.exportStackDialog.depthStackMinDepthInput.value() / 1000, self.exportStackDialog.depthStackMaxDepthInput.value() / 1000)
                     nDepths = int(self.exportStackDialog.depthStackNumDepthsInput.value())
                     QApplication.setOverrideCursor(Qt.WaitCursor)
                     depthStack = self.imageProcessor.holo.depth_stack(self.imageProcessor.preProcessFrame, depthRange, nDepths)
                     QApplication.restoreOverrideCursor()
                     depthStack.write_intensity_to_tif(filename)
        else:
              QMessageBox.about(self, "Error", "A hologram is required to create a depth stack.") 



class ExportStackDialog(QDialog):
    """ Dialog box that appears when export depth stack is clicked."
    """
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Export Stack")
        self.setMinimumWidth(300)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.depthStackMinDepthInput = QDoubleSpinBox()
        self.depthStackMinDepthInput.setMaximum(10**6)
        self.depthStackMinDepthInput.setValue(0)
        self.depthStackMaxDepthInput = QDoubleSpinBox()
        self.depthStackMaxDepthInput.setMaximum(10**6)
        self.depthStackMaxDepthInput.setValue(1)

        self.depthStackNumDepthsInput = QSpinBox()
        self.depthStackNumDepthsInput.setMaximum(10**6)
        self.depthStackNumDepthsInput.setValue(10)

        self.layout.addWidget(QLabel("Start Depth (mm):"))
        self.layout.addWidget(self.depthStackMinDepthInput)
        self.layout.addWidget(QLabel("End Depth (mm):"))
        self.layout.addWidget(self.depthStackMaxDepthInput)
        self.layout.addWidget(QLabel("Number of Depths:"))
        self.layout.addWidget(self.depthStackNumDepthsInput)
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        

        
        
if __name__ == '__main__':    
    
    app=QApplication(sys.argv)
           
    window=Holo_Bundle()
    window.show()
    sys.exit(app.exec_())

