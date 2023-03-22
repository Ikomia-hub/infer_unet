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

        # image size
        self.spin_size = pyqtutils.append_spin(self.gridLayout, "img_size", self.parameters.img_size)

        # MODEL FILE
        self.browse_model_file = pyqtutils.append_browse_file(grid_layout=self.gridLayout, label="model File",
                                                              path=self.parameters.modelFile,
                                                              mode=QFileDialog.ExistingFile)

        # Set widget layout
        self.set_layout(layout)

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.img_size = self.spin_size.value()
        self.parameters.modelFile = self.browse_model_file.path

        # Send signal to launch the process
        self.emit_apply(self.parameters)


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
