# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'newCNC.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1039, 720)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QSize(926, 720))
        MainWindow.setLocale(QLocale(QLocale.English, QLocale.UnitedKingdom))
        self.actionLoad = QAction(MainWindow)
        self.actionLoad.setObjectName(u"actionLoad")
        font = QFont()
        font.setBold(True)
        font.setWeight(75)
        self.actionLoad.setFont(font)
        self.actionHide_Show_Console = QAction(MainWindow)
        self.actionHide_Show_Console.setObjectName(u"actionHide_Show_Console")
        self.actionHide_Show_Console.setCheckable(True)
        self.actionHide_Show_Console.setChecked(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.verticalLayout_6 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setFont(font)
        self.viewTab = QWidget()
        self.viewTab.setObjectName(u"viewTab")
        self.viewTab.setFont(font)
        self.horizontalLayout_8 = QHBoxLayout(self.viewTab)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetFixedSize)
        self.drillFileLineEdit = QLineEdit(self.viewTab)
        self.drillFileLineEdit.setObjectName(u"drillFileLineEdit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.drillFileLineEdit.sizePolicy().hasHeightForWidth())
        self.drillFileLineEdit.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.drillFileLineEdit, 4, 0, 1, 1)

        self.checkBox_4 = QCheckBox(self.viewTab)
        self.checkBox_4.setObjectName(u"checkBox_4")
        sizePolicy1.setHeightForWidth(self.checkBox_4.sizePolicy().hasHeightForWidth())
        self.checkBox_4.setSizePolicy(sizePolicy1)
        self.checkBox_4.setChecked(True)

        self.gridLayout.addWidget(self.checkBox_4, 5, 2, 1, 1, Qt.AlignHCenter)

        self.pushButton_3 = QPushButton(self.viewTab)
        self.pushButton_3.setObjectName(u"pushButton_3")
        sizePolicy1.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.pushButton_3, 6, 1, 1, 1)

        self.drillViewCheckBox = QCheckBox(self.viewTab)
        self.drillViewCheckBox.setObjectName(u"drillViewCheckBox")
        sizePolicy1.setHeightForWidth(self.drillViewCheckBox.sizePolicy().hasHeightForWidth())
        self.drillViewCheckBox.setSizePolicy(sizePolicy1)
        self.drillViewCheckBox.setChecked(True)

        self.gridLayout.addWidget(self.drillViewCheckBox, 4, 2, 1, 1, Qt.AlignHCenter)

        self.lineEdit_3 = QLineEdit(self.viewTab)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        sizePolicy1.setHeightForWidth(self.lineEdit_3.sizePolicy().hasHeightForWidth())
        self.lineEdit_3.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.lineEdit_3, 6, 0, 1, 1)

        self.profileFileLineEdit = QLineEdit(self.viewTab)
        self.profileFileLineEdit.setObjectName(u"profileFileLineEdit")
        sizePolicy1.setHeightForWidth(self.profileFileLineEdit.sizePolicy().hasHeightForWidth())
        self.profileFileLineEdit.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.profileFileLineEdit, 3, 0, 1, 1)

        self.topFileLineEdit = QLineEdit(self.viewTab)
        self.topFileLineEdit.setObjectName(u"topFileLineEdit")
        sizePolicy1.setHeightForWidth(self.topFileLineEdit.sizePolicy().hasHeightForWidth())
        self.topFileLineEdit.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.topFileLineEdit, 1, 0, 1, 1)

        self.topLoadButton = QPushButton(self.viewTab)
        self.topLoadButton.setObjectName(u"topLoadButton")
        sizePolicy1.setHeightForWidth(self.topLoadButton.sizePolicy().hasHeightForWidth())
        self.topLoadButton.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.topLoadButton, 1, 1, 1, 1)

        self.bottomLoadButton = QPushButton(self.viewTab)
        self.bottomLoadButton.setObjectName(u"bottomLoadButton")
        sizePolicy1.setHeightForWidth(self.bottomLoadButton.sizePolicy().hasHeightForWidth())
        self.bottomLoadButton.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.bottomLoadButton, 2, 1, 1, 1)

        self.filePathLabel = QLabel(self.viewTab)
        self.filePathLabel.setObjectName(u"filePathLabel")
        sizePolicy1.setHeightForWidth(self.filePathLabel.sizePolicy().hasHeightForWidth())
        self.filePathLabel.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.filePathLabel, 0, 0, 1, 1, Qt.AlignHCenter)

        self.bottomViewCheckBox = QCheckBox(self.viewTab)
        self.bottomViewCheckBox.setObjectName(u"bottomViewCheckBox")
        sizePolicy1.setHeightForWidth(self.bottomViewCheckBox.sizePolicy().hasHeightForWidth())
        self.bottomViewCheckBox.setSizePolicy(sizePolicy1)
        self.bottomViewCheckBox.setChecked(True)

        self.gridLayout.addWidget(self.bottomViewCheckBox, 2, 2, 1, 1, Qt.AlignHCenter)

        self.profileViewCheckBox = QCheckBox(self.viewTab)
        self.profileViewCheckBox.setObjectName(u"profileViewCheckBox")
        sizePolicy1.setHeightForWidth(self.profileViewCheckBox.sizePolicy().hasHeightForWidth())
        self.profileViewCheckBox.setSizePolicy(sizePolicy1)
        self.profileViewCheckBox.setChecked(True)

        self.gridLayout.addWidget(self.profileViewCheckBox, 3, 2, 1, 1, Qt.AlignHCenter)

        self.pushButton_2 = QPushButton(self.viewTab)
        self.pushButton_2.setObjectName(u"pushButton_2")
        sizePolicy1.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.pushButton_2, 5, 1, 1, 1)

        self.viewLabel = QLabel(self.viewTab)
        self.viewLabel.setObjectName(u"viewLabel")
        sizePolicy1.setHeightForWidth(self.viewLabel.sizePolicy().hasHeightForWidth())
        self.viewLabel.setSizePolicy(sizePolicy1)
        self.viewLabel.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.viewLabel, 0, 2, 1, 1, Qt.AlignHCenter)

        self.bottomFileLineEdit = QLineEdit(self.viewTab)
        self.bottomFileLineEdit.setObjectName(u"bottomFileLineEdit")
        sizePolicy1.setHeightForWidth(self.bottomFileLineEdit.sizePolicy().hasHeightForWidth())
        self.bottomFileLineEdit.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.bottomFileLineEdit, 2, 0, 1, 1)

        self.checkBox_5 = QCheckBox(self.viewTab)
        self.checkBox_5.setObjectName(u"checkBox_5")
        sizePolicy1.setHeightForWidth(self.checkBox_5.sizePolicy().hasHeightForWidth())
        self.checkBox_5.setSizePolicy(sizePolicy1)
        self.checkBox_5.setChecked(True)

        self.gridLayout.addWidget(self.checkBox_5, 6, 2, 1, 1, Qt.AlignHCenter)

        self.drillLoadButton = QPushButton(self.viewTab)
        self.drillLoadButton.setObjectName(u"drillLoadButton")
        sizePolicy1.setHeightForWidth(self.drillLoadButton.sizePolicy().hasHeightForWidth())
        self.drillLoadButton.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.drillLoadButton, 4, 1, 1, 1)

        self.lineEdit_2 = QLineEdit(self.viewTab)
        self.lineEdit_2.setObjectName(u"lineEdit_2")
        sizePolicy1.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.lineEdit_2, 5, 0, 1, 1)

        self.profileLoadButton = QPushButton(self.viewTab)
        self.profileLoadButton.setObjectName(u"profileLoadButton")
        sizePolicy1.setHeightForWidth(self.profileLoadButton.sizePolicy().hasHeightForWidth())
        self.profileLoadButton.setSizePolicy(sizePolicy1)

        self.gridLayout.addWidget(self.profileLoadButton, 3, 1, 1, 1)

        self.topViewCheckBox = QCheckBox(self.viewTab)
        self.topViewCheckBox.setObjectName(u"topViewCheckBox")
        sizePolicy1.setHeightForWidth(self.topViewCheckBox.sizePolicy().hasHeightForWidth())
        self.topViewCheckBox.setSizePolicy(sizePolicy1)
        self.topViewCheckBox.setChecked(True)
        self.topViewCheckBox.setTristate(False)

        self.gridLayout.addWidget(self.topViewCheckBox, 1, 2, 1, 1, Qt.AlignHCenter)


        self.verticalLayout_2.addLayout(self.gridLayout)

        self.verticalSpacer = QSpacerItem(20, 448, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)


        self.horizontalLayout_6.addLayout(self.verticalLayout_2)

        self.viewCanvasWidget = QOpenGLWidget(self.viewTab)
        self.viewCanvasWidget.setObjectName(u"viewCanvasWidget")
        sizePolicy.setHeightForWidth(self.viewCanvasWidget.sizePolicy().hasHeightForWidth())
        self.viewCanvasWidget.setSizePolicy(sizePolicy)

        self.horizontalLayout_6.addWidget(self.viewCanvasWidget)


        self.horizontalLayout_8.addLayout(self.horizontalLayout_6)

        self.tabWidget.addTab(self.viewTab, "")
        self.controlTab = QWidget()
        self.controlTab.setObjectName(u"controlTab")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.controlTab.sizePolicy().hasHeightForWidth())
        self.controlTab.setSizePolicy(sizePolicy2)
        self.controlTab.setFont(font)
        self.horizontalLayout_5 = QHBoxLayout(self.controlTab)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.controlsVerticalLayout = QVBoxLayout()
        self.controlsVerticalLayout.setObjectName(u"controlsVerticalLayout")
        self.controlsVerticalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.statusLabel = QLabel(self.controlTab)
        self.statusLabel.setObjectName(u"statusLabel")
        sizePolicy3 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.statusLabel.sizePolicy().hasHeightForWidth())
        self.statusLabel.setSizePolicy(sizePolicy3)
        font1 = QFont()
        font1.setPointSize(12)
        font1.setBold(True)
        font1.setWeight(75)
        self.statusLabel.setFont(font1)
        self.statusLabel.setAlignment(Qt.AlignCenter)

        self.controlsVerticalLayout.addWidget(self.statusLabel)

        self.droGridLayout = QGridLayout()
        self.droGridLayout.setObjectName(u"droGridLayout")
        self.droGridLayout.setHorizontalSpacing(0)
        self.droGridLayout.setVerticalSpacing(6)
        self.zAxisLabel = QLabel(self.controlTab)
        self.zAxisLabel.setObjectName(u"zAxisLabel")
        sizePolicy3.setHeightForWidth(self.zAxisLabel.sizePolicy().hasHeightForWidth())
        self.zAxisLabel.setSizePolicy(sizePolicy3)
        self.zAxisLabel.setFont(font)
        self.zAxisLabel.setLayoutDirection(Qt.LeftToRight)
        self.zAxisLabel.setFrameShape(QFrame.NoFrame)
        self.zAxisLabel.setFrameShadow(QFrame.Plain)
        self.zAxisLabel.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.zAxisLabel, 0, 3, 1, 1)

        self.label_10 = QLabel(self.controlTab)
        self.label_10.setObjectName(u"label_10")
        sizePolicy3.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy3)
        self.label_10.setLayoutDirection(Qt.LeftToRight)
        self.label_10.setFrameShape(QFrame.NoFrame)
        self.label_10.setFrameShadow(QFrame.Plain)
        self.label_10.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.label_10, 2, 1, 1, 1)

        self.zero_z_pushButton = QPushButton(self.controlTab)
        self.zero_z_pushButton.setObjectName(u"zero_z_pushButton")
        self.zero_z_pushButton.setFont(font)

        self.droGridLayout.addWidget(self.zero_z_pushButton, 3, 3, 1, 1)

        self.zero_xyz_pushButton = QPushButton(self.controlTab)
        self.zero_xyz_pushButton.setObjectName(u"zero_xyz_pushButton")
        self.zero_xyz_pushButton.setFont(font)

        self.droGridLayout.addWidget(self.zero_xyz_pushButton, 3, 0, 1, 1)

        self.label_9 = QLabel(self.controlTab)
        self.label_9.setObjectName(u"label_9")
        sizePolicy3.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy3)
        self.label_9.setLayoutDirection(Qt.LeftToRight)
        self.label_9.setFrameShape(QFrame.NoFrame)
        self.label_9.setFrameShadow(QFrame.Plain)
        self.label_9.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.label_9, 2, 2, 1, 1)

        self.label_11 = QLabel(self.controlTab)
        self.label_11.setObjectName(u"label_11")
        sizePolicy3.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy3)
        self.label_11.setLayoutDirection(Qt.LeftToRight)
        self.label_11.setFrameShape(QFrame.NoFrame)
        self.label_11.setFrameShadow(QFrame.Plain)
        self.label_11.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.label_11, 2, 3, 1, 1)

        self.zero_y_pushButton = QPushButton(self.controlTab)
        self.zero_y_pushButton.setObjectName(u"zero_y_pushButton")
        self.zero_y_pushButton.setFont(font)

        self.droGridLayout.addWidget(self.zero_y_pushButton, 3, 2, 1, 1)

        self.xAxisLabel = QLabel(self.controlTab)
        self.xAxisLabel.setObjectName(u"xAxisLabel")
        sizePolicy3.setHeightForWidth(self.xAxisLabel.sizePolicy().hasHeightForWidth())
        self.xAxisLabel.setSizePolicy(sizePolicy3)
        self.xAxisLabel.setFont(font)
        self.xAxisLabel.setLayoutDirection(Qt.LeftToRight)
        self.xAxisLabel.setFrameShape(QFrame.NoFrame)
        self.xAxisLabel.setFrameShadow(QFrame.Plain)
        self.xAxisLabel.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.xAxisLabel, 0, 1, 1, 1)

        self.zero_x_pushButton = QPushButton(self.controlTab)
        self.zero_x_pushButton.setObjectName(u"zero_x_pushButton")
        self.zero_x_pushButton.setFont(font)

        self.droGridLayout.addWidget(self.zero_x_pushButton, 3, 1, 1, 1)

        self.xAxisLabel_2 = QLabel(self.controlTab)
        self.xAxisLabel_2.setObjectName(u"xAxisLabel_2")
        sizePolicy3.setHeightForWidth(self.xAxisLabel_2.sizePolicy().hasHeightForWidth())
        self.xAxisLabel_2.setSizePolicy(sizePolicy3)
        self.xAxisLabel_2.setFont(font)
        self.xAxisLabel_2.setLayoutDirection(Qt.LeftToRight)
        self.xAxisLabel_2.setFrameShape(QFrame.NoFrame)
        self.xAxisLabel_2.setFrameShadow(QFrame.Plain)
        self.xAxisLabel_2.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.xAxisLabel_2, 0, 0, 1, 1)

        self.yAxisLabel = QLabel(self.controlTab)
        self.yAxisLabel.setObjectName(u"yAxisLabel")
        sizePolicy3.setHeightForWidth(self.yAxisLabel.sizePolicy().hasHeightForWidth())
        self.yAxisLabel.setSizePolicy(sizePolicy3)
        self.yAxisLabel.setFont(font)
        self.yAxisLabel.setLayoutDirection(Qt.LeftToRight)
        self.yAxisLabel.setFrameShape(QFrame.NoFrame)
        self.yAxisLabel.setFrameShadow(QFrame.Plain)
        self.yAxisLabel.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.yAxisLabel, 0, 2, 1, 1)

        self.label_12 = QLabel(self.controlTab)
        self.label_12.setObjectName(u"label_12")
        sizePolicy3.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy3)
        self.label_12.setFont(font)
        self.label_12.setLayoutDirection(Qt.LeftToRight)
        self.label_12.setFrameShape(QFrame.NoFrame)
        self.label_12.setFrameShadow(QFrame.Plain)
        self.label_12.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.label_12, 2, 0, 1, 1)

        self.mpos_y_label = QLabel(self.controlTab)
        self.mpos_y_label.setObjectName(u"mpos_y_label")
        sizePolicy3.setHeightForWidth(self.mpos_y_label.sizePolicy().hasHeightForWidth())
        self.mpos_y_label.setSizePolicy(sizePolicy3)
        self.mpos_y_label.setLayoutDirection(Qt.LeftToRight)
        self.mpos_y_label.setFrameShape(QFrame.NoFrame)
        self.mpos_y_label.setFrameShadow(QFrame.Plain)
        self.mpos_y_label.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.mpos_y_label, 1, 2, 1, 1)

        self.mpos_x_label = QLabel(self.controlTab)
        self.mpos_x_label.setObjectName(u"mpos_x_label")
        sizePolicy3.setHeightForWidth(self.mpos_x_label.sizePolicy().hasHeightForWidth())
        self.mpos_x_label.setSizePolicy(sizePolicy3)
        self.mpos_x_label.setLayoutDirection(Qt.LeftToRight)
        self.mpos_x_label.setFrameShape(QFrame.NoFrame)
        self.mpos_x_label.setFrameShadow(QFrame.Plain)
        self.mpos_x_label.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.mpos_x_label, 1, 1, 1, 1)

        self.mpos_z_label = QLabel(self.controlTab)
        self.mpos_z_label.setObjectName(u"mpos_z_label")
        sizePolicy3.setHeightForWidth(self.mpos_z_label.sizePolicy().hasHeightForWidth())
        self.mpos_z_label.setSizePolicy(sizePolicy3)
        self.mpos_z_label.setLayoutDirection(Qt.LeftToRight)
        self.mpos_z_label.setFrameShape(QFrame.NoFrame)
        self.mpos_z_label.setFrameShadow(QFrame.Plain)
        self.mpos_z_label.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.mpos_z_label, 1, 3, 1, 1)

        self.label_5 = QLabel(self.controlTab)
        self.label_5.setObjectName(u"label_5")
        sizePolicy3.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy3)
        self.label_5.setFont(font)
        self.label_5.setLayoutDirection(Qt.LeftToRight)
        self.label_5.setFrameShape(QFrame.NoFrame)
        self.label_5.setFrameShadow(QFrame.Plain)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.droGridLayout.addWidget(self.label_5, 1, 0, 1, 1)


        self.controlsVerticalLayout.addLayout(self.droGridLayout)

        self.unlockHomingHorizontalLayout = QHBoxLayout()
        self.unlockHomingHorizontalLayout.setObjectName(u"unlockHomingHorizontalLayout")
        self.unlockHomingHorizontalLayout.setContentsMargins(20, 20, 20, 20)
        self.unlockButton = QPushButton(self.controlTab)
        self.unlockButton.setObjectName(u"unlockButton")
        sizePolicy1.setHeightForWidth(self.unlockButton.sizePolicy().hasHeightForWidth())
        self.unlockButton.setSizePolicy(sizePolicy1)
        self.unlockButton.setFont(font)

        self.unlockHomingHorizontalLayout.addWidget(self.unlockButton)

        self.homingButton = QPushButton(self.controlTab)
        self.homingButton.setObjectName(u"homingButton")
        sizePolicy1.setHeightForWidth(self.homingButton.sizePolicy().hasHeightForWidth())
        self.homingButton.setSizePolicy(sizePolicy1)
        self.homingButton.setFont(font)

        self.unlockHomingHorizontalLayout.addWidget(self.homingButton)


        self.controlsVerticalLayout.addLayout(self.unlockHomingHorizontalLayout)

        self.gridLayoutDirections = QGridLayout()
        self.gridLayoutDirections.setObjectName(u"gridLayoutDirections")
        self.gridLayoutDirections.setContentsMargins(5, 5, 5, 5)
        self.xYMinusPlusButton = QPushButton(self.controlTab)
        self.xYMinusPlusButton.setObjectName(u"xYMinusPlusButton")
        sizePolicy1.setHeightForWidth(self.xYMinusPlusButton.sizePolicy().hasHeightForWidth())
        self.xYMinusPlusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.xYMinusPlusButton, 0, 0, 1, 1)

        self.yPlusButton = QPushButton(self.controlTab)
        self.yPlusButton.setObjectName(u"yPlusButton")
        sizePolicy1.setHeightForWidth(self.yPlusButton.sizePolicy().hasHeightForWidth())
        self.yPlusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.yPlusButton, 0, 1, 1, 1)

        self.xYPlusButton = QPushButton(self.controlTab)
        self.xYPlusButton.setObjectName(u"xYPlusButton")
        sizePolicy1.setHeightForWidth(self.xYPlusButton.sizePolicy().hasHeightForWidth())
        self.xYPlusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.xYPlusButton, 0, 2, 1, 1)

        self.xMinusButton = QPushButton(self.controlTab)
        self.xMinusButton.setObjectName(u"xMinusButton")
        sizePolicy1.setHeightForWidth(self.xMinusButton.sizePolicy().hasHeightForWidth())
        self.xMinusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.xMinusButton, 1, 0, 1, 1)

        self.centerButton = QPushButton(self.controlTab)
        self.centerButton.setObjectName(u"centerButton")
        sizePolicy1.setHeightForWidth(self.centerButton.sizePolicy().hasHeightForWidth())
        self.centerButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.centerButton, 1, 1, 1, 1)

        self.xPlusButton = QPushButton(self.controlTab)
        self.xPlusButton.setObjectName(u"xPlusButton")
        sizePolicy1.setHeightForWidth(self.xPlusButton.sizePolicy().hasHeightForWidth())
        self.xPlusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.xPlusButton, 1, 2, 1, 1)

        self.xYMinusButton = QPushButton(self.controlTab)
        self.xYMinusButton.setObjectName(u"xYMinusButton")
        sizePolicy1.setHeightForWidth(self.xYMinusButton.sizePolicy().hasHeightForWidth())
        self.xYMinusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.xYMinusButton, 2, 0, 1, 1)

        self.yMinusButton = QPushButton(self.controlTab)
        self.yMinusButton.setObjectName(u"yMinusButton")
        sizePolicy1.setHeightForWidth(self.yMinusButton.sizePolicy().hasHeightForWidth())
        self.yMinusButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.yMinusButton, 2, 1, 1, 1)

        self.xYPlusMinuButton = QPushButton(self.controlTab)
        self.xYPlusMinuButton.setObjectName(u"xYPlusMinuButton")
        sizePolicy1.setHeightForWidth(self.xYPlusMinuButton.sizePolicy().hasHeightForWidth())
        self.xYPlusMinuButton.setSizePolicy(sizePolicy1)

        self.gridLayoutDirections.addWidget(self.xYPlusMinuButton, 2, 2, 1, 1)


        self.controlsVerticalLayout.addLayout(self.gridLayoutDirections)

        self.xyzIncrementsHorizontalLayout = QHBoxLayout()
        self.xyzIncrementsHorizontalLayout.setObjectName(u"xyzIncrementsHorizontalLayout")
        self.xyzIncrementsHorizontalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_3 = QLabel(self.controlTab)
        self.label_3.setObjectName(u"label_3")
        sizePolicy1.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy1)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_3)

        self.doubleSpinBox_2 = QDoubleSpinBox(self.controlTab)
        self.doubleSpinBox_2.setObjectName(u"doubleSpinBox_2")
        sizePolicy1.setHeightForWidth(self.doubleSpinBox_2.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_2.setSizePolicy(sizePolicy1)

        self.verticalLayout_5.addWidget(self.doubleSpinBox_2)


        self.xyzIncrementsHorizontalLayout.addLayout(self.verticalLayout_5)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_4 = QLabel(self.controlTab)
        self.label_4.setObjectName(u"label_4")
        sizePolicy1.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy1)
        self.label_4.setAlignment(Qt.AlignCenter)

        self.verticalLayout_4.addWidget(self.label_4)

        self.doubleSpinBox = QDoubleSpinBox(self.controlTab)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        sizePolicy1.setHeightForWidth(self.doubleSpinBox.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox.setSizePolicy(sizePolicy1)

        self.verticalLayout_4.addWidget(self.doubleSpinBox)


        self.xyzIncrementsHorizontalLayout.addLayout(self.verticalLayout_4)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.zPlusButton = QPushButton(self.controlTab)
        self.zPlusButton.setObjectName(u"zPlusButton")
        sizePolicy1.setHeightForWidth(self.zPlusButton.sizePolicy().hasHeightForWidth())
        self.zPlusButton.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.zPlusButton)

        self.zMinusButton = QPushButton(self.controlTab)
        self.zMinusButton.setObjectName(u"zMinusButton")
        sizePolicy1.setHeightForWidth(self.zMinusButton.sizePolicy().hasHeightForWidth())
        self.zMinusButton.setSizePolicy(sizePolicy1)

        self.verticalLayout_3.addWidget(self.zMinusButton)


        self.xyzIncrementsHorizontalLayout.addLayout(self.verticalLayout_3)

        self.xyzIncrementsHorizontalLayout.setStretch(0, 3)
        self.xyzIncrementsHorizontalLayout.setStretch(1, 3)
        self.xyzIncrementsHorizontalLayout.setStretch(2, 1)

        self.controlsVerticalLayout.addLayout(self.xyzIncrementsHorizontalLayout)

        self.terminalVerticalLayout = QVBoxLayout()
        self.terminalVerticalLayout.setObjectName(u"terminalVerticalLayout")
        self.textEdit = QTextEdit(self.controlTab)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setEnabled(True)
        self.textEdit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.terminalVerticalLayout.addWidget(self.textEdit)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.send_text_edit = QLineEdit(self.controlTab)
        self.send_text_edit.setObjectName(u"send_text_edit")

        self.horizontalLayout_2.addWidget(self.send_text_edit)

        self.send_push_button = QPushButton(self.controlTab)
        self.send_push_button.setObjectName(u"send_push_button")
        sizePolicy4 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.send_push_button.sizePolicy().hasHeightForWidth())
        self.send_push_button.setSizePolicy(sizePolicy4)
        self.send_push_button.setFont(font)

        self.horizontalLayout_2.addWidget(self.send_push_button)


        self.terminalVerticalLayout.addLayout(self.horizontalLayout_2)

        self.connectVerticalLayout = QVBoxLayout()
        self.connectVerticalLayout.setSpacing(6)
        self.connectVerticalLayout.setObjectName(u"connectVerticalLayout")
        self.connectVerticalLayout.setContentsMargins(5, 2, 5, 2)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.serialPortsComboBox = QComboBox(self.controlTab)
        self.serialPortsComboBox.setObjectName(u"serialPortsComboBox")
        sizePolicy5 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.serialPortsComboBox.sizePolicy().hasHeightForWidth())
        self.serialPortsComboBox.setSizePolicy(sizePolicy5)

        self.horizontalLayout.addWidget(self.serialPortsComboBox)

        self.refreshButton = QPushButton(self.controlTab)
        self.refreshButton.setObjectName(u"refreshButton")
        sizePolicy4.setHeightForWidth(self.refreshButton.sizePolicy().hasHeightForWidth())
        self.refreshButton.setSizePolicy(sizePolicy4)
        self.refreshButton.setFont(font)
        self.refreshButton.setCheckable(False)

        self.horizontalLayout.addWidget(self.refreshButton)


        self.connectVerticalLayout.addLayout(self.horizontalLayout)


        self.terminalVerticalLayout.addLayout(self.connectVerticalLayout)

        self.splitter_2 = QSplitter(self.controlTab)
        self.splitter_2.setObjectName(u"splitter_2")
        self.splitter_2.setOrientation(Qt.Horizontal)
        self.checkBox = QCheckBox(self.splitter_2)
        self.checkBox.setObjectName(u"checkBox")
        sizePolicy6 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy6)
        self.checkBox.setFont(font)
        self.checkBox.setChecked(True)
        self.splitter_2.addWidget(self.checkBox)
        self.clearTerminalButton = QPushButton(self.splitter_2)
        self.clearTerminalButton.setObjectName(u"clearTerminalButton")
        self.clearTerminalButton.setEnabled(True)
        sizePolicy6.setHeightForWidth(self.clearTerminalButton.sizePolicy().hasHeightForWidth())
        self.clearTerminalButton.setSizePolicy(sizePolicy6)
        self.clearTerminalButton.setFont(font)
        self.splitter_2.addWidget(self.clearTerminalButton)
        self.connectButton = QPushButton(self.splitter_2)
        self.connectButton.setObjectName(u"connectButton")
        sizePolicy1.setHeightForWidth(self.connectButton.sizePolicy().hasHeightForWidth())
        self.connectButton.setSizePolicy(sizePolicy1)
        self.connectButton.setFont(font)
        self.splitter_2.addWidget(self.connectButton)

        self.terminalVerticalLayout.addWidget(self.splitter_2)


        self.controlsVerticalLayout.addLayout(self.terminalVerticalLayout)


        self.horizontalLayout_4.addLayout(self.controlsVerticalLayout)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.openGLWidget = QOpenGLWidget(self.controlTab)
        self.openGLWidget.setObjectName(u"openGLWidget")
        sizePolicy.setHeightForWidth(self.openGLWidget.sizePolicy().hasHeightForWidth())
        self.openGLWidget.setSizePolicy(sizePolicy)

        self.verticalLayout.addWidget(self.openGLWidget)


        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 10)

        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)

        self.tabWidget.addTab(self.controlTab, "")
        self.alignTab = QWidget()
        self.alignTab.setObjectName(u"alignTab")
        self.alignTab.setFont(font)
        self.horizontalLayout_7 = QHBoxLayout(self.alignTab)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.openGLWidget_2 = QOpenGLWidget(self.alignTab)
        self.openGLWidget_2.setObjectName(u"openGLWidget_2")

        self.horizontalLayout_7.addWidget(self.openGLWidget_2)

        self.label_2 = QLabel(self.alignTab)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout_7.addWidget(self.label_2)

        self.verticalSlider = QSlider(self.alignTab)
        self.verticalSlider.setObjectName(u"verticalSlider")
        self.verticalSlider.setMaximum(255)
        self.verticalSlider.setOrientation(Qt.Vertical)

        self.horizontalLayout_7.addWidget(self.verticalSlider)

        self.tabWidget.addTab(self.alignTab, "")

        self.verticalLayout_6.addWidget(self.tabWidget)

        self.consoleTextEdit = QTextEdit(self.centralwidget)
        self.consoleTextEdit.setObjectName(u"consoleTextEdit")

        self.verticalLayout_6.addWidget(self.consoleTextEdit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1039, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuFile.setFont(font)
        self.menuConsole = QMenu(self.menubar)
        self.menuConsole.setObjectName(u"menuConsole")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuConsole.menuAction())
        self.menuFile.addAction(self.actionLoad)
        self.menuConsole.addAction(self.actionHide_Show_Console)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"newCNC", None))
        self.actionLoad.setText(QCoreApplication.translate("MainWindow", u"Open", None))
#if QT_CONFIG(shortcut)
        self.actionLoad.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.actionHide_Show_Console.setText(QCoreApplication.translate("MainWindow", u"Hide/Show Console", None))
#if QT_CONFIG(shortcut)
        self.actionHide_Show_Console.setShortcut(QCoreApplication.translate("MainWindow", u"Shift+C", None))
#endif // QT_CONFIG(shortcut)
        self.checkBox_4.setText("")
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.drillViewCheckBox.setText("")
        self.topLoadButton.setText(QCoreApplication.translate("MainWindow", u"TOP", None))
        self.bottomLoadButton.setText(QCoreApplication.translate("MainWindow", u"BOTTOM", None))
        self.filePathLabel.setText(QCoreApplication.translate("MainWindow", u"File Path", None))
        self.bottomViewCheckBox.setText("")
        self.profileViewCheckBox.setText("")
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.viewLabel.setText(QCoreApplication.translate("MainWindow", u"View", None))
        self.checkBox_5.setText("")
        self.drillLoadButton.setText(QCoreApplication.translate("MainWindow", u"DRILL", None))
        self.profileLoadButton.setText(QCoreApplication.translate("MainWindow", u"PROFILE", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.viewTab), QCoreApplication.translate("MainWindow", u"VIEW", None))
        self.statusLabel.setText(QCoreApplication.translate("MainWindow", u"Not Connected", None))
        self.zAxisLabel.setText(QCoreApplication.translate("MainWindow", u"Z", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.zero_z_pushButton.setText(QCoreApplication.translate("MainWindow", u"Z = 0", None))
        self.zero_xyz_pushButton.setText(QCoreApplication.translate("MainWindow", u"XYZ = 0", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.zero_y_pushButton.setText(QCoreApplication.translate("MainWindow", u"Y = 0", None))
        self.xAxisLabel.setText(QCoreApplication.translate("MainWindow", u"X", None))
        self.zero_x_pushButton.setText(QCoreApplication.translate("MainWindow", u"X = 0", None))
        self.xAxisLabel_2.setText(QCoreApplication.translate("MainWindow", u"Axis:", None))
        self.yAxisLabel.setText(QCoreApplication.translate("MainWindow", u"Y", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"WPos", None))
        self.mpos_y_label.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.mpos_x_label.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.mpos_z_label.setText(QCoreApplication.translate("MainWindow", u"0.0", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"MPos", None))
        self.unlockButton.setText(QCoreApplication.translate("MainWindow", u"Unlock", None))
        self.homingButton.setText(QCoreApplication.translate("MainWindow", u"Homing", None))
        self.xYMinusPlusButton.setText(QCoreApplication.translate("MainWindow", u"X- Y+", None))
        self.yPlusButton.setText(QCoreApplication.translate("MainWindow", u"Y+", None))
        self.xYPlusButton.setText(QCoreApplication.translate("MainWindow", u"X+ Y+", None))
        self.xMinusButton.setText(QCoreApplication.translate("MainWindow", u"X-", None))
        self.centerButton.setText(QCoreApplication.translate("MainWindow", u"Center", None))
        self.xPlusButton.setText(QCoreApplication.translate("MainWindow", u"X+", None))
        self.xYMinusButton.setText(QCoreApplication.translate("MainWindow", u"X- Y-", None))
        self.yMinusButton.setText(QCoreApplication.translate("MainWindow", u"Y-", None))
        self.xYPlusMinuButton.setText(QCoreApplication.translate("MainWindow", u"X+ Y-", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"XY", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Z", None))
        self.zPlusButton.setText(QCoreApplication.translate("MainWindow", u"Z+", None))
        self.zMinusButton.setText(QCoreApplication.translate("MainWindow", u"Z-", None))
        self.send_push_button.setText(QCoreApplication.translate("MainWindow", u"Send", None))
#if QT_CONFIG(tooltip)
        self.refreshButton.setToolTip(QCoreApplication.translate("MainWindow", u"Refresh serial port list.", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.refreshButton.setStatusTip(QCoreApplication.translate("MainWindow", u"Refresh serial port list.", None))
#endif // QT_CONFIG(statustip)
        self.refreshButton.setText(QCoreApplication.translate("MainWindow", u"Serial Port Refresh", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Autoscroll", None))
        self.clearTerminalButton.setText(QCoreApplication.translate("MainWindow", u"Clear Terminal", None))
#if QT_CONFIG(tooltip)
        self.connectButton.setToolTip(QCoreApplication.translate("MainWindow", u"Connect to selected serial port.", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(statustip)
        self.connectButton.setStatusTip(QCoreApplication.translate("MainWindow", u"Connect to selected serial port.", None))
#endif // QT_CONFIG(statustip)
        self.connectButton.setText(QCoreApplication.translate("MainWindow", u"Connect", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.controlTab), QCoreApplication.translate("MainWindow", u"CONTROL", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.alignTab), QCoreApplication.translate("MainWindow", u"ALIGN", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuConsole.setTitle(QCoreApplication.translate("MainWindow", u"Console", None))
    # retranslateUi

