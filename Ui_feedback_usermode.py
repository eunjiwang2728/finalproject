# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\CODING_SPACE\final\feedbacksystem\feedback-usermode.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_userWindow(object):
    def setupUi(self, userWindow):
        userWindow.setObjectName("userWindow")
        userWindow.setEnabled(True)
        userWindow.resize(1126, 732)
        font = QtGui.QFont()
        font.setFamily("GimhaeGaya Bold")
        userWindow.setFont(font)
        userWindow.setCursor(QtGui.QCursor(QtCore.Qt.SizeFDiagCursor))
        userWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(userWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setContentsMargins(30, 30, 30, 30)
        self.gridLayout_2.setHorizontalSpacing(20)
        self.gridLayout_2.setVerticalSpacing(10)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.user_camera_layout = QtWidgets.QVBoxLayout()
        self.user_camera_layout.setSpacing(10)
        self.user_camera_layout.setObjectName("user_camera_layout")
        self.label_status = QtWidgets.QLabel(self.centralwidget)
        self.label_status.setAutoFillBackground(False)
        self.label_status.setStyleSheet("background-color: rgb(87, 127, 49);")
        self.label_status.setText("")
        self.label_status.setAlignment(QtCore.Qt.AlignCenter)
        self.label_status.setObjectName("label_status")
        self.user_camera_layout.addWidget(self.label_status)
        self.user_cameraframe = QtWidgets.QGraphicsView(self.centralwidget)
        self.user_cameraframe.setObjectName("user_cameraframe")
        self.user_camera_layout.addWidget(self.user_cameraframe)
        self.button_layout = QtWidgets.QSplitter(self.centralwidget)
        self.button_layout.setOrientation(QtCore.Qt.Horizontal)
        self.button_layout.setObjectName("button_layout")
        self.layoutWidget = QtWidgets.QWidget(self.button_layout)
        self.layoutWidget.setObjectName("layoutWidget")
        self.button_sublayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.button_sublayout.setContentsMargins(0, 0, 0, 0)
        self.button_sublayout.setObjectName("button_sublayout")
        self.user_button_expression = QtWidgets.QPushButton(self.layoutWidget)
        self.user_button_expression.setStyleSheet("background: rgb(141, 212, 212);\n"
"font: 11pt \"GimhaeGaya Bold\";")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/img/imgsource/happy.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.user_button_expression.setIcon(icon)
        self.user_button_expression.setObjectName("user_button_expression")
        self.button_sublayout.addWidget(self.user_button_expression, 0, 1, 1, 1)
        self.user_button_voice = QtWidgets.QPushButton(self.layoutWidget)
        self.user_button_voice.setStyleSheet("background: rgb(85, 170, 127);\n"
"font: 11pt \"GimhaeGaya Bold\";")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/img/imgsource/microphone-black-shape.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.user_button_voice.setIcon(icon1)
        self.user_button_voice.setObjectName("user_button_voice")
        self.button_sublayout.addWidget(self.user_button_voice, 0, 0, 1, 1)
        self.user_button_gaze = QtWidgets.QPushButton(self.layoutWidget)
        self.user_button_gaze.setStyleSheet("background-color: rgb(255, 170, 127);\n"
"font: 11pt \"GimhaeGaya Bold\";")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/img/imgsource/eye.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.user_button_gaze.setIcon(icon2)
        self.user_button_gaze.setObjectName("user_button_gaze")
        self.button_sublayout.addWidget(self.user_button_gaze, 0, 2, 1, 1)
        self.user_camera_layout.addWidget(self.button_layout)
        self.user_camera_layout.setStretch(0, 1)
        self.user_camera_layout.setStretch(1, 12)
        self.user_camera_layout.setStretch(2, 1)

        self.gridLayout_2.addLayout(self.user_camera_layout, 0, 0, 4, 1)

        self.ratioButton_layout = QtWidgets.QHBoxLayout()
        self.ratioButton_layout.setObjectName("ratioButton_layout")
        self.radioButton_user = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_user.setChecked(True)
        self.radioButton_user.setObjectName("radioButton_user")
        self.ratioButton_layout.addWidget(self.radioButton_user)
        self.radioButton_test = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_test.setEnabled(False)
        self.radioButton_test.setObjectName("radioButton_test")
        self.ratioButton_layout.addWidget(self.radioButton_test)
        self.gridLayout_2.addLayout(self.ratioButton_layout, 0, 1, 1, 1)

        self.avatar = QtWidgets.QGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.avatar.sizePolicy().hasHeightForWidth())
        self.avatar.setSizePolicy(sizePolicy)
        self.avatar.setMinimumSize(QtCore.QSize(101, 0))
        self.avatar.setSizeIncrement(QtCore.QSize(2, 2))
        self.avatar.setStyleSheet("background-position: center; background-repeat: norepeat;\n"
"background-image: url(:/img/imgsource/avatar.png);")
        self.avatar.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.avatar.setLineWidth(0)
        self.avatar.setObjectName("avatar")
        self.gridLayout_2.addWidget(self.avatar, 1, 1, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.user_ui_feedbacktext = QtWidgets.QLabel(self.centralwidget)
        self.user_ui_feedbacktext.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.user_ui_feedbacktext.sizePolicy().hasHeightForWidth())
        self.user_ui_feedbacktext.setSizePolicy(sizePolicy)
        self.user_ui_feedbacktext.setStyleSheet("font: 20pt \"GimhaeGaya\"; color: rgb(85, 170, 255); background-color: white; ")
        self.user_ui_feedbacktext.setAlignment(QtCore.Qt.AlignCenter)
        self.user_ui_feedbacktext.setObjectName("user_ui_feedbacktext")
        self.gridLayout_2.addWidget(self.user_ui_feedbacktext, 2, 1, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setHorizontalSpacing(15)
        self.gridLayout.setVerticalSpacing(10)
        self.gridLayout.setObjectName("gridLayout")
        self.icon_face = QtWidgets.QLabel(self.centralwidget)
        self.icon_face.setStyleSheet("image: url(:/img/imgsource/happy.png);")
        self.icon_face.setText("")
        self.icon_face.setObjectName("icon_face")
        self.gridLayout.addWidget(self.icon_face, 2, 0, 1, 1)
        self.icon_gaze = QtWidgets.QLabel(self.centralwidget)
        self.icon_gaze.setStyleSheet("image: url(:/img/imgsource/eye.png);")
        self.icon_gaze.setText("")
        self.icon_gaze.setObjectName("icon_gaze")
        self.gridLayout.addWidget(self.icon_gaze, 1, 0, 1, 1)
        self.label_feedback_gaze = QtWidgets.QLabel(self.centralwidget)
        self.label_feedback_gaze.setStyleSheet("font: 20pt \"GimhaeGaya Bold\";\n"
"background-color: rgb(255, 255, 255);")
        self.label_feedback_gaze.setAlignment(QtCore.Qt.AlignCenter)
        self.label_feedback_gaze.setWordWrap(True)
        self.label_feedback_gaze.setObjectName("label_feedback_gaze")
        self.gridLayout.addWidget(self.label_feedback_gaze, 1, 2, 1, 1)
        self.label_feedback_voice = QtWidgets.QLabel(self.centralwidget)
        self.label_feedback_voice.setStyleSheet("font: 20pt \"GimhaeGaya Bold\";\n"
"background-color: rgb(255, 255, 255);")
        self.label_feedback_voice.setAlignment(QtCore.Qt.AlignCenter)
        self.label_feedback_voice.setWordWrap(True)
        self.label_feedback_voice.setObjectName("label_feedback_voice")
        self.gridLayout.addWidget(self.label_feedback_voice, 0, 2, 1, 1)
        self.label_feedback_expression = QtWidgets.QLabel(self.centralwidget)
        self.label_feedback_expression.setStyleSheet("font: 20pt \"GimhaeGaya Bold\";\n"
"background-color: rgb(255, 255, 255);")
        self.label_feedback_expression.setAlignment(QtCore.Qt.AlignCenter)
        self.label_feedback_expression.setWordWrap(True)
        self.label_feedback_expression.setObjectName("label_feedback_expression")
        self.gridLayout.addWidget(self.label_feedback_expression, 2, 2, 1, 1)
        self.icon_voice = QtWidgets.QLabel(self.centralwidget)
        self.icon_voice.setStyleSheet("image: url(:/img/imgsource/microphone-black-shape.png);")
        self.icon_voice.setText("")
        self.icon_voice.setObjectName("icon_voice")
        self.gridLayout.addWidget(self.icon_voice, 0, 0, 1, 1)
        self.alert_face = QtWidgets.QLabel(self.centralwidget)
        self.alert_face.setText("")
        self.alert_face.setObjectName("alert_face")
        self.gridLayout.addWidget(self.alert_face, 2, 1, 1, 1)
        self.alert_gaze = QtWidgets.QLabel(self.centralwidget)
        self.alert_gaze.setText("")
        self.alert_gaze.setObjectName("alert_gaze")
        self.gridLayout.addWidget(self.alert_gaze, 1, 1, 1, 1)
        self.alert_voice = QtWidgets.QLabel(self.centralwidget)
        self.alert_voice.setText("")
        self.alert_voice.setObjectName("alert_voice")
        self.gridLayout.addWidget(self.alert_voice, 0, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 5)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 20)
        self.gridLayout_2.addLayout(self.gridLayout, 3, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_2.setColumnStretch(1, 1)
        self.gridLayout_2.setRowStretch(0, 1)
        self.gridLayout_2.setRowStretch(1, 1)
        self.gridLayout_2.setRowStretch(2, 1)
        self.gridLayout_2.setRowStretch(3, 10)
        userWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(userWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1126, 23))
        self.menubar.setObjectName("menubar")
        userWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(userWindow)
        self.statusbar.setObjectName("statusbar")
        userWindow.setStatusBar(self.statusbar)
        self.action_menu_user_mode = QtWidgets.QAction(userWindow)
        self.action_menu_user_mode.setObjectName("action_menu_user_mode")
        self.actionrealtime_mode = QtWidgets.QAction(userWindow)
        self.actionrealtime_mode.setObjectName("actionrealtime_mode")
        self.actionvideo_mode = QtWidgets.QAction(userWindow)
        self.actionvideo_mode.setObjectName("actionvideo_mode")
        self.action_menu_video_mode = QtWidgets.QAction(userWindow)
        self.action_menu_video_mode.setObjectName("action_menu_video_mode")
        self.action_menu_realtime_test_mode = QtWidgets.QAction(userWindow)
        self.action_menu_realtime_test_mode.setObjectName("action_menu_realtime_test_mode")

        self.retranslateUi(userWindow)
        QtCore.QMetaObject.connectSlotsByName(userWindow)

    def retranslateUi(self, userWindow):
        _translate = QtCore.QCoreApplication.translate
        userWindow.setWindowTitle(_translate("userWindow", "모의화상 면접 시스템"))
        self.radioButton_user.setText(_translate("userWindow", "user"))
        self.radioButton_test.setText(_translate("userWindow", "test"))
        self.user_button_expression.setText(_translate("userWindow", " 표정"))
        self.user_button_voice.setText(_translate("userWindow", "음성"))
        self.user_button_gaze.setText(_translate("userWindow", "시선"))
        self.user_ui_feedbacktext.setText(_translate("userWindow", "피드백"))
        self.label_feedback_gaze.setText(_translate("userWindow", "시선"))
        self.label_feedback_voice.setText(_translate("userWindow", "음성"))
        self.label_feedback_expression.setText(_translate("userWindow", "표정"))
        self.action_menu_user_mode.setText(_translate("userWindow", "user mode"))
        self.action_menu_user_mode.setShortcut(_translate("userWindow", "M"))
        self.actionrealtime_mode.setText(_translate("userWindow", "realtime mode"))
        self.actionvideo_mode.setText(_translate("userWindow", "video mode"))
        self.action_menu_video_mode.setText(_translate("userWindow", "video mode"))
        self.action_menu_realtime_test_mode.setText(_translate("userWindow", "realtime test mode"))
import pictures_rc

