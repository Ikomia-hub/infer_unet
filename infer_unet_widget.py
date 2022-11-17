# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_unet.infer_unet_process import InferUnetParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferUnetWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferUnetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        # PyQt -> Qt wrapping
        layout = qtconversion.PyQtToQt(self.gridLayout)

        img_scaleLabel = QLabel("Scale (resize images)")
        self.img_scaleSpinBox = QDoubleSpinBox()
        self.img_scaleSpinBox.setRange(0.1, 1)
        self.img_scaleSpinBox.setDecimals(4)
        self.img_scaleSpinBox.setSingleStep(0.0001)
        self.img_scaleSpinBox.setValue(self.parameters.img_scale)

        num_classesLabel = QLabel("Classes number:")
        self.num_classesSpinBox = QSpinBox()
        self.num_classesSpinBox.setRange(1, 2147483647)
        self.num_classesSpinBox.setSingleStep(1)
        self.num_classesSpinBox.setValue(self.parameters.num_classes)

        num_channelsLabel = QLabel("Channels number:")
        self.num_channelsSpinBox = QSpinBox()
        self.num_channelsSpinBox.setRange(1, 4)
        self.num_channelsSpinBox.setSingleStep(1)
        self.num_channelsSpinBox.setValue(self.parameters.num_channels)

        # Set widget layout
        self.gridLayout.addWidget(img_scaleLabel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.img_scaleSpinBox, 0, 1, 1, 2)
        self.gridLayout.addWidget(num_classesLabel, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.num_classesSpinBox, 1, 1, 1, 2)
        self.gridLayout.addWidget(num_channelsLabel, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.num_channelsSpinBox, 2, 1, 1, 2)

        # MODEL FILE
        self.browse_model_file = pyqtutils.append_browse_file(grid_layout=self.gridLayout, label="model File",
                                                              path=self.parameters.modelFile,
                                                              mode=QFileDialog.ExistingFile)
        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.gridLayout, label="Output folder",
                                                              path=self.parameters.outputFolder,
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        # Set widget layout
        self.setLayout(layout)

    def onApply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.img_scale = self.img_scaleSpinBox.value()
        self.parameters.num_classes = self.num_classesSpinBox.value()
        self.parameters.num_channels = self.num_channelsSpinBox.value()
        self.parameters.outputFolder = self.browse_out_folder.path
        self.parameters.modelFile = self.browse_model_file.path

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferUnetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_unet"

    def create(self, param):
        # Create widget object
        return InferUnetWidget(param, None)
