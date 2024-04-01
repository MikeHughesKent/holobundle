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
This file handles only the holography specific elements of processing and 
creating the holography processing panel and the refocusing panel. 
Super-resolution GUI elements are also implemented here.

@author: Mike Hughes, Applied Optics Group, Physics and Astronomy, University of Kent


"""

import sys 
import os
from pathlib import Path
import time
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt

import serial
from PIL import Image
import cv2 as cv

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPalette, QColor, QImage, QPixmap, QPainter, QPen, QGuiApplication
from PyQt5.QtGui import QPainter, QBrush, QPen

file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '..\..\cas\src')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '..\..\pyholoscope\src')))
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '..\..\pyfibrebundle\src')))


from cas_gui.base import CAS_GUI
from cas_gui.subclasses.cas_bundle import CAS_GUI_Bundle

from processors.inline_bundle_processor_class import InlineBundleProcessorClass

import pyholoscope

import pybundle
from pybundle import PyBundle
from pybundle import SuperRes

# Led Modes
SEQUENTIAL = 0
SINGLE = 1
resPath = "../../cas/res"

class Holo_Bundle(CAS_GUI_Bundle):
    
    authorName = "AOG"
    appName = "HoloBundle"
    windowTitle = "HoloBundle"
    logoFilename = "../res/kent_logo_2.png"
    resPath = "../../cas/res"
    
    processor = InlineBundleProcessorClass
    
    multiCore = True   
    sharedMemory = True
    cuda = True    
    srBackgrounds = None   
    sr = True
    mosaicingEnabled = False
    sr_single_led_id = 1  
    serial = None
    
    def __init__(self,parent=None):        
        
        super(Holo_Bundle, self).__init__(parent)        
        

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
        self.sourceFilename = r"C:\Users\mrh40\Dropbox\Programming\Python\holoBundle\tests\test_data\sr_test_1.tif"
     
        self.rawImageBufferSize = 20

        # Call these functions to update things based on the default GUI options
        self.sr_clear_shifts_clicked()
        self.handle_sr_enabled()
        
        self.exportStackDialog = ExportStackDialog()
        try:
            self.load_background()
        except:
            pass
        
        try:
            self.load_calibration()
        except:
            pass
        

 
    
    def create_layout(self):
        """ Called by parent class to assemble the GUI from Qt Widgets"""
        
        
        super().create_layout()
        
        # Create the additional menu buttons needed for HoloBundle
        self.holoMenuButton = self.create_menu_button("Holography Settings", QIcon('../res/icons/disc_white.svg'), self.holo_menu_button_clicked, True, True, 7)
        self.stackButton = self.create_menu_button("Depth Stack", QIcon('../res/icons/copy_white.svg'), self.depth_stack_clicked, False, False, 5)
        self.srMenuButton = self.create_menu_button("Resolution Enhancement", QIcon('../res/icons/layers_white.svg'), self.sr_menu_button_clicked, True, True, 9)

        # Create the additional menu panels needed for HoloBundle
        self.holoPanel = self.create_inline_holo_panel()
        self.srPanel = self.create_inline_holo_sr_panel()
        
        
        # We rename the bundle handling menus to distinguish them from the holography menus
        self.settingsButton.setText(" Bundle Settings")      
        self.calibMenuButton.setText(" Bundle Calibration")   
        
        # Create the long depth slider
        self.create_focus_panel()
        
        

    def create_focus_panel(self): 
    
        """ Create a long slider for focusing
        """
        self.holoDepthInput = QDoubleSpinBox(objectName='holoDepthInput')
        self.holoDepthInput.setKeyboardTracking(False)
        self.holoDepthInput.setMaximum(10**6)
        self.holoDepthInput.setMinimum(-10**6)
        self.holoDepthInput.setSingleStep(10)
        self.holoDepthInput.setMinimumWidth(90)
        self.holoDepthInput.setMaximumWidth(90)
        self.longFocusWidget = QWidget(objectName = "long_focus")
        self.longFocusWidget.setContentsMargins(0,0,0,0)
        self.longFocusWidgetLayout = QVBoxLayout()
        self.longFocusWidget.setLayout(self.longFocusWidgetLayout)
        self.longFocusWidget.setMinimumWidth(190)
        self.longFocusWidget.setMaximumWidth(190)
        
        
        self.holoLongDepthSlider = QSlider(QtCore.Qt.Vertical, objectName = 'longHoloDepthSlider')
        self.holoLongDepthSlider.setInvertedAppearance(True)
        self.refocusTitle = QLabel("Refocus")
        self.longFocusWidgetLayout.addWidget(self.refocusTitle, alignment=QtCore.Qt.AlignHCenter)
        self.refocusTitle.setProperty("subheader", "true")
        self.refocusTitle.setStyleSheet("QLabel{padding:5px}")
        self.holoLongDepthSlider.setStyleSheet("QSlider{padding:20px}")
        self.longFocusWidgetLayout.addWidget(self.holoLongDepthSlider, alignment=QtCore.Qt.AlignHCenter)
        self.contentLayout.addWidget(self.longFocusWidget) 
        self.longFocusWidgetLayout.addWidget(QLabel('Depth, \u03bcm'),alignment=QtCore.Qt.AlignHCenter)
        self.longFocusWidgetLayout.addWidget(self.holoDepthInput,alignment=QtCore.Qt.AlignHCenter)  
        self.longFocusWidget.setStyleSheet("QWidget{padding:0px; margin:0px;background-color:rgba(30, 30, 60, 255)}")
        self.holoDepthInput.setMaximumWidth(90)

        self.holoDepthInput.valueChanged[float].connect(self.holo_depth_changed)
        self.holoDepthInput.setStyleSheet("QDoubleSpinBox{padding: 5px; background-color: rgba(255, 255, 255, 255); color: black; font-size:9pt}")
        self.holoLongDepthSlider.valueChanged[int].connect(self.long_depth_slider_changed)       
        self.holoLongDepthSlider.setTickPosition(QSlider.TicksBelow)
        self.holoLongDepthSlider.setTickInterval(100)
        self.holoLongDepthSlider.setMaximum(5000)
        
        file = "../res/holo_bundle.css"        
        with open(file,"r") as fh:
            self.holoLongDepthSlider.setStyleSheet(fh.read())
      
  

    def create_inline_holo_panel(self):
        """ Create the panel with controls for holography"""   
        
        widget, layout = self.panel_helper(title = "Holography Settings")
        
        self.holoRefocusCheck = QCheckBox("Refocus", objectName='holoRefocusCheck')
        self.holoDifferentialCheck = QCheckBox("Differential", objectName='holoDifferentialCheck')
        self.holoPhaseCheck = QCheckBox("Show Phase", objectName='holoPhaseCheck')
        self.holoInvertCheck = QCheckBox("Invert Image", objectName='holoInvertCheck')
        
        self.holoWavelengthInput = QDoubleSpinBox(objectName='holoWavelengthInput')
        self.holoWavelengthInput.setMaximum(10**6)
        self.holoWavelengthInput.setMinimum(-10**6)
        self.holoWavelengthInput.setDecimals(3)
        
        self.holoPixelSizeInput = QDoubleSpinBox(objectName='holoPixelSizeInput')
        self.holoPixelSizeInput.setMaximum(10**6)
        self.holoPixelSizeInput.setMinimum(-10**6)
        
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
      
        layout.addWidget(self.holoRefocusCheck)
        layout.addWidget(self.holoPhaseCheck)
        layout.addWidget(self.holoInvertCheck)    
        layout.addWidget(self.holoDifferentialCheck)           
        
        layout.addWidget(QLabel('Wavelegnth (microns):'))
        layout.addWidget(self.holoWavelengthInput)
        
        layout.addWidget(QLabel('Pixel Size (microns):'))
        layout.addWidget(self.holoPixelSizeInput) 
        
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
        
        layout.addWidget(QLabel("Adjusted Pixel Size (microns):"))
        self.adjustedPixelSizeLabel = QLabel("")
        layout.addWidget(self.adjustedPixelSizeLabel)
        self.adjustedPixelSizeLabel.setProperty('status', 'true')

        layout.addStretch()
        
        # We call processing_options_changed because we have overloaded this from CAS_GUI_Bundle and
        # its convenient to have one functions that handles all updates to processing
        self.holoWavelengthInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoPixelSizeInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoRefocusCheck.stateChanged.connect(self.processing_options_changed)
        self.holoPhaseCheck.stateChanged.connect(self.processing_options_changed)
        self.holoDifferentialCheck.stateChanged.connect(self.processing_options_changed)
        self.holoInvertCheck.stateChanged.connect(self.processing_options_changed)
        self.holoWindowThicknessInput.valueChanged[float].connect(self.processing_options_changed)
        self.holoWindowCombo.currentIndexChanged[int].connect(self.processing_options_changed)
        self.holoSliderMaxInput.valueChanged[int].connect(self.processing_options_changed)

        return widget  


    def create_inline_holo_sr_panel(self):
        """Create the panel for super-resolution"""
        
        widget, layout = self.panel_helper(title = "Resolution Enhancement")

        
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
         
        layout.addWidget(self.srEnabledCheck)

        layout.addWidget(self.srCalibBtn)
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
        
        layout.addStretch()
      
        self.srCalibBtn.clicked.connect(self.sr_calibrate_click)
        self.srEnabledCheck.stateChanged.connect(self.handle_sr_enabled)
        self.srSaveCalibBtn.clicked.connect(self.save_sr_calib_clicked)
        self.srLoadCalibBtn.clicked.connect(self.load_sr_calib_clicked)
        self.srAcquireBackgroundsBtn.clicked.connect(self.acquire_sr_background_clicked)
        self.srSaveBackgroundsBtn.clicked.connect(self.save_sr_background_clicked)
        self.srLoadBackgroundsBtn.clicked.connect(self.load_sr_background_clicked)
        self.srMultiBackgroundsCheck.stateChanged.connect(self.processing_options_changed)
        self.srMultiNormalisationCheck.stateChanged.connect(self.processing_options_changed)
        self.srUseLUTCheck.stateChanged.connect(self.processing_options_changed)
        
        self.holoWindowThicknessInput.valueChanged[float].connect(self.processing_options_changed)
        self.srGenerateLUTBtn.clicked.connect(self.sr_generate_LUT_clicked)
        self.srCaptureShiftBtn.clicked.connect(self.sr_capture_shift_clicked)
        
        self.srSaveCalibrationLUTBtn.clicked.connect(self.sr_save_calibration_lut_clicked)
        self.srLoadCalibrationLUTBtn.clicked.connect(self.sr_load_calibration_lut_clicked)
        
        return widget  
    
   
    def long_depth_slider_changed(self):
        self.holoDepthInput.setValue(int(self.holoLongDepthSlider.value()))      
    
    def holo_menu_button_clicked(self):
        self.expanding_menu_clicked(self.holoMenuButton, self.holoPanel)
        
    def sr_menu_button_clicked(self):
        self.expanding_menu_clicked(self.srMenuButton, self.srPanel)    
    
    
    def handle_plot_button(self, handle_depth_slider):
        fig, axs = plt.subplots(2, 4, dpi=150)
        axs[0,0].imshow(self.imageProcessor.currentInputImage[:,:,0])
        axs[0,1].imshow(self.imageProcessor.currentInputImage[:,:,1])
        axs[0,2].imshow(self.imageProcessor.currentInputImage[:,:,2])
        axs[0,3].imshow(self.imageProcessor.currentInputImage[:,:,3])
        axs[1,0].imshow(self.imageProcessor.currentInputImage[:,:,4])
        axs[1,1].imshow(self.imageProcessor.currentInputImage[:,:,5])
        axs[1,2].imshow(self.imageProcessor.currentInputImage[:,:,6])

        
  
        
    

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
                   
                   
    
    def processing_options_changed(self):   
        """Called when changes are made to the options pane. Updates the processor thread."""
        
        # Check the slider matches the input box value and the max value is correct
        self.holoLongDepthSlider.setValue(int(self.holoDepthInput.value()))
        self.holoLongDepthSlider.setMaximum(int(self.holoSliderMaxInput.value()))

        # Holography specific processing
        if self.imageProcessor is not None:
            self.imageProcessor.get_processor().showPhase = self.holoPhaseCheck.isChecked()
            self.imageProcessor.get_processor().invert = self.holoInvertCheck.isChecked()
            self.imageProcessor.get_processor().holo.cuda = self.cuda   
            if self.holoRefocusCheck.isChecked():
                
                self.imageProcessor.get_processor().refocus = True
                
                # Wavelength
                if self.holoWavelengthInput.value() != self.imageProcessor.get_processor().holo.wavelength / 10**6:
                    self.imageProcessor.get_processor().holo.set_wavelength(self.holoWavelengthInput.value()/ 10**6)
                
                # Depth
                if self.holoDepthInput.value() != self.imageProcessor.get_processor().holo.depth / 10**6:
                    self.imageProcessor.get_processor().holo.set_depth(self.holoDepthInput.value()/ 10**6)
                
                # Windowing
                if self.holoWindowCombo.currentText() == "Circular":
                    self.imageProcessor.get_processor().holo.set_auto_window(True)
                    self.imageProcessor.get_processor().holo.set_window_shape('circle')                    
                    self.imageProcessor.get_processor().holo.set_window_radius(None)
                    self.imageProcessor.get_processor().holo.set_window_thickness(self.holoWindowThicknessInput.value())
                else:
                    self.imageProcessor.get_processor().holo.window = None
            else:
                self.imageProcessor.get_processor().refocus = False
        
        # The basic bundle processing is defined in CAS_GUI_Bundle
        super().processing_options_changed()
       
        # Have to do pixel size last in case bundle processing changed as this
        # may change scale factor. If we changed the target pixel size we then also
        # have to call update_file_processing again, which was called by 
        # processing_options_changed in the parent class _before_ we
        # updated the pixel size
        
        if self.imageProcessor is not None:
            scaleFactor = self.imageProcessor.get_processor().pyb.get_pixel_scale() 
            if scaleFactor is not None:
                targetPixelSize = self.imageProcessor.get_processor().pyb.get_pixel_scale() * self.holoPixelSizeInput.value() / 10**6
            else:
                targetPixelSize = self.holoPixelSizeInput.value() / 10**6
            self.adjustedPixelSizeLabel.setText("Adjusted Pixel Size: " + str(round(targetPixelSize * 10**6,2)) + "microns" )

            if targetPixelSize != self.imageProcessor.get_processor().holo.pixelSize:
                self.imageProcessor.get_processor().holo.set_pixel_size(targetPixelSize)
                self.update_file_processing()
            
            else:
                self.imageProcessor.set_batch_process_num(1)
                self.imageProcessor.get_processor().set_differential(False)
                
            if self.srEnabledCheck.isChecked():
                
                self.imageProcessor.get_processor().sr = True
                self.imageProcessor.get_processor().pyb.set_super_res(True)
                self.imageProcessor.get_processor().pyb.set_sr_backgrounds(self.srBackgrounds)
                self.imageProcessor.get_processor().pyb.set_sr_normalisation_images(self.srBackgrounds)
                self.imageProcessor.get_processor().pyb.set_sr_multi_normalisation(self.srMultiNormalisationCheck.isChecked())
                self.imageProcessor.get_processor().pyb.set_sr_multi_backgrounds(self.srMultiBackgroundsCheck.isChecked())
                self.imageProcessor.get_processor().set_batch_process_num(self.srNumShiftsInput.value() + 1)
                self.imageProcessor.get_processor().pyb.set_sr_use_lut(self.srUseLUTCheck.isChecked())
                self.imageProcessor.get_processor().pyb.set_sr_param_value(self.holoDepthInput.value()/ 10**6)

                if self.imageThread is not None:
                    self.imageThread.set_num_removal_when_full(self.srNumShiftsInput.value() + 1)
                self.update_file_processing()
                
            elif self.holoDifferentialCheck.isChecked():
                self.imageProcessor.set_batch_process_num(2)
                self.imageProcessor.get_processor().set_differential(True)
                self.imageProcessor.get_processor().sr = False
                self.imageProcessor.get_processor().pyb.set_super_res(False)
                self.update_file_processing()

            else:
                self.imageProcessor.get_processor().sr = False
                self.imageProcessor.get_processor().pyb.set_super_res(False)
                self.imageProcessor.set_batch_process_num(1)
                self.imageProcessor.get_processor().set_differential(False)
                self.update_file_processing()

            self.imageProcessor.update_settings()


    def holo_depth_changed(self):
        if self.imageProcessor is not None:
            if self.holoDepthInput.value() != self.imageProcessor.get_processor().holo.depth / 10**6:
                self.imageProcessor.get_processor().holo.set_depth(self.holoDepthInput.value()/ 10**6)
                self.update_file_processing()
                self.imageProcessor.pipe_message('set_depth', self.holoDepthInput.value()/ 10**6)


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
          self.processing_options_changed()    
      
      
    def sr_set_led_mode(self, mode):
          """ If using an array of LEDs, communicate with Arduino to set correct operation 
          for current mode.
          
          """
          if self.serial is not None:
              self.serial.reset_output_buffer()
      
              if mode == SEQUENTIAL:
                  self.serial.write(b'm')
              if mode == SINGLE:
                  self.serial.write(('s' + str(self.sr_single_led_id) + '\n').encode('utf_8'))   
                  
      
    
    def auto_focus_clicked(self):
        """ Finds best focus and updates depth slider
        """
     
        if self.mainDisplay.roi is not None:
            roi = pyholoscope.Roi(self.mainDisplay.roi[0], self.mainDisplay.roi[1], self.even(self.mainDisplay.roi[2] - self.mainDisplay.roi[0]), self.even(self.mainDisplay.roi[3] - self.mainDisplay.roi[1]))
        else:
            roi = None
        autofocusMax = self.holoAutoFocusMaxInput.value() / 1000
        autofocusMin = self.holoAutoFocusMinInput.value() / 1000
        if self.holoAutoFocusCoarseDivisionsInput.value() > 1:
            numSearchDivisions = int(self.holoAutoFocusCoarseDivisionsInput.value())
        else:
            numSearchDivisions = None
        autofocusROIMargin = self.holoAutoFocusROIMarginInput.value()
        if self.imageThread is not None:
            self.imageThread.pause()
        autoFocus = (self.imageProcessor.auto_focus(roi = roi, method = 'Peak', margin = None, depthRange = (autofocusMin, autofocusMax), coarseSearchInterval =  numSearchDivisions))
        self.holoDepthInput.setValue(autoFocus * 1000) 

        if self.imageThread is not None:
            self.imageThread.resume()

        
    def sr_generate_LUT_clicked(self):
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
        
        
    def sr_capture_shift_clicked(self):
        """
        """
        if self.imageProcessor.pyb.calibration is None:
            QMessageBox.about(self, "Error", "Shift measurement requires an interpolation calibration.")  
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.sr_param_holograms.append(self.imageProcessor.get_processor().capture_sr_shift())
        self.sr_param_depths.append(self.imageProcessor.get_processor().holo.depth)
        QApplication.restoreOverrideCursor()

   
    def sr_clear_shifts_clicked(self):
        """
        """
        self.sr_param_holograms = []
        self.sr_param_depths = []


    def sr_save_calibration_lut_clicked(self):  
        with open('sr_calib_lut.dat','wb') as pickleFile:
            pickle.dump(self.imageProcessor.pyb.srCalibrationLUT, pickleFile)
        
    def sr_load_calibration_lut_clicked(self):  
        with open('sr_calib_lut.dat', 'rb') as pickleFile:
            self.imageProcessor.pyb.srCalibrationLUT = pickle.load(pickleFile)
        self.processing_options_changed()  
        
        
    def sr_calibrate_click(self): 
        """ Called when SR Calibrate is clicked
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
        
        self.processing_options_changed()   
        
            
    def save_sr_calib_clicked(self):
        """ Saves SR calibration to a file.
        """
        with open('sr_calib.dat','wb') as pickleFile:
            pickle.dump(self.imageProcessor.pyb.calibrationSR, pickleFile)

        
    def load_sr_calib_clicked(self):
        """ Loads SR calibration from a file.
        """
        with open('sr_calib.dat', 'rb') as pickleFile:
            self.imageProcessor.pyb.calibrationSR = pickle.load(pickleFile)
        self.processing_options_changed()        
        
        
    def save_sr_background_clicked(self):
        """ If we have a stack of backgrounds for SR, save to a TIF stack.
        """
        if self.srBackgrounds is not None:
            imlist = []

            for idx in range(np.shape(self.srBackgrounds)[2]):
                imlist.append(Image.fromarray(self.srBackgrounds[:,:,idx].astype('uint16')))

            imlist[0].save('sr_backgrounds.tif', compression="tiff_deflate", save_all=True,
                   append_images=imlist[1:])                   

        
    def load_sr_background_clicked(self):
        """ Loads a tif stack of images and sets as current super-resolution backgrounds stack.
        """
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
        self.processing_options_changed()
        
        
    def acquire_sr_background_clicked(self):
        """ Sets the current SR image stack as the background stack and updates processor
        """
        if self.imageProcessor is not None and self.srEnabledCheck.isChecked():
            self.srBackgrounds = pybundle.SuperRes.sort_sr_stack(self.imageProcessor.currentInputImage, self.imageProcessor.batchProcessNum - 1)    
            self.backgroundImage = self.srBackgrounds[:,:,self.sr_single_led_id]
            self.processing_options_changed()
            
            
    def apply_default_settings(self):
        """Applies hard-coded default settings. These will then be saved on program
        exit. """
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
            
            
    def depth_stack_clicked(self):
        """ Creates a depth stack over a specified range.
        """
        
        if self.imageProcessor is not None and (self.imageProcessor.preProcessFrame is not None or self.currentImage is not None):
            if self.exportStackDialog.exec():
                try:
                    filename = QFileDialog.getSaveFileName(self, 'Select filename to save to:', '', filter='*.tif')[0]
                except:
                    filename = None
                if filename is not None and filename != '':
                     depthRange = (self.exportStackDialog.depthStackMinDepthInput.value() / 1000, self.exportStackDialog.depthStackMaxDepthInput.value() / 1000)
                     nDepths = int(self.exportStackDialog.depthStackNumDepthsInput.value())
                     QApplication.setOverrideCursor(Qt.WaitCursor)
                     depthStack = self.imageProcessor.get_processor().holo.depth_stack(self.imageProcessor.preProcessFrame, depthRange, nDepths)
                     QApplication.restoreOverrideCursor()
                     depthStack.write_intensity_to_tif(filename)
        else:
              QMessageBox.about(self, "Error", "A hologram is required to create a depth stack.") 



class ExportStackDialog(QDialog):
    """ Dialog box that appears when export depth stack is clicked."
    """
    
    def __init__(self):
        super().__init__()
        
        file=os.path.join(resPath, 'cas_modern.css')
        with open(file,"r") as fh:
            self.setStyleSheet(fh.read())

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

