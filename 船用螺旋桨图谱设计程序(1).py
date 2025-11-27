import sys
import csv
import math
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
                             QGroupBox, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QTableWidget, QTableWidgetItem, QTextEdit, QHBoxLayout,
                             QFileDialog, QMessageBox, QGridLayout, QRadioButton,
                             QDialog, QDialogButtonBox, QSpinBox, QDoubleSpinBox, QComboBox,
                             QFrame, QSizePolicy, QSpacerItem)
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QFontDatabase
from PyQt5.QtCore import Qt, QSize
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, Akima1DInterpolator, CubicSpline
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 解决matplotlib中文显示问题
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 10  # 设置全局字体大小

# ---------- 全局常量 ----------
SIGMA_WAG = [0.1136, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.488]
TAU_C_WAG = [0.0777, 0.135, 0.1582, 0.1846, 0.206, 0.2304, 0.2633, 0.2876, 0.34]
SIGMA_BER = [0.36, 0.389, 0.407, 0.416, 0.481, 0.54, 0.6, 0.7, 0.806, 0.834, 0.848, 0.9, 1.82]
TAU_C_BER = [0.14, 0.162, 0.164, 0.169, 0.175, 0.190, 0.200, 0.223, 0.224, 0.227, 0.228, 0.251, 0.35]

MAU_THICKNESS = {'0.2R': 4.06, '0.3R': 3.59, '0.4R': 3.12, '0.5R': 2.65,
                 '0.6R': 2.18, '0.7R': 1.71, '0.8R': 1.24, '0.9R': 0.77, '1.0R': 0.30}
MAU_WIDTH = {'0.2R': 66.54, '0.3R': 77.70, '0.4R': 87.08, '0.5R': 94.34,
             '0.6R': 99.11, '0.7R': 99.64, '0.8R': 92.92, '0.9R': 73.62, '1.0R': 0.0}
SIMPSON_COEFF = {'0.2R': 1, '0.3R': 4, '0.4R': 2, '0.5R': 4, '0.6R': 2,
                 '0.7R': 4, '0.8R': 2, '0.9R': 4, '1.0R': 1}
AREA_COEFF = {'0.2R': 0.674, '0.3R': 0.674, '0.4R': 0.674, '0.5R': 0.6745,
              '0.6R': 0.6745, '0.7R': 0.677, '0.8R': 0.683, '0.9R': 0.695, '1.0R': 0.700}


# MAU螺旋桨系数管理类
class AUCoefficients:
    """AU螺旋桨系数管理类"""

    def __init__(self):
        # 4叶桨KT系数表
        self.kt_coeffs_4 = [
            {'value': -0.2536277E-01, 'i': 0, 'j': 0, 'k': 0},
            {'value': -0.2072556E+00, 'i': 0, 'j': 1, 'k': 0},
            {'value': 0.5724472E+00, 'i': 1, 'j': 0, 'k': 0},
            {'value': 0.1939063E+00, 'i': 2, 'j': 0, 'k': 3},
            {'value': -0.2890781E+00, 'i': 0, 'j': 2, 'k': 2},
            {'value': -0.1074432E+01, 'i': 1, 'j': 2, 'k': 2},
            {'value': -0.2131741E+00, 'i': 2, 'j': 0, 'k': 0},
            {'value': 0.2703334E+00, 'i': 2, 'j': 0, 'k': 1},
            {'value': 0.1870137E-01, 'i': 3, 'j': 1, 'k': 0},
            {'value': 0.9646077E+00, 'i': 0, 'j': 3, 'k': 3},
            {'value': -0.2029306E+00, 'i': 0, 'j': 4, 'k': 3},
            {'value': 0.1305797E-02, 'i': 7, 'j': 0, 'k': 1},
            {'value': -0.5234681E-01, 'i': 0, 'j': 0, 'k': 1},
            {'value': -0.1710635E+00, 'i': 0, 'j': 2, 'k': 0},
            {'value': 0.7317558E+00, 'i': 1, 'j': 2, 'k': 1},
            {'value': -0.1049158E+00, 'i': 1, 'j': 0, 'k': 2},
            {'value': 0.6117029E-01, 'i': 5, 'j': 1, 'k': 3},
            {'value': -0.1214246E+00, 'i': 0, 'j': 3, 'k': 1},
            {'value': -0.5872456E-02, 'i': 7, 'j': 2, 'k': 1},
            {'value': -0.1525986E+00, 'i': 1, 'j': 1, 'k': 1},
            {'value': 0.1006423E-02, 'i': 7, 'j': 4, 'k': 1},
            {'value': -0.8940443E-01, 'i': 4, 'j': 0, 'k': 3}
        ]

        # 4叶桨KQ系数表
        self.kq_coeffs_4 = [
            {'value': 0.3899004E-01, 'i': 0, 'j': 0, 'k': 0},
            {'value': 0.2886616E+00, 'i': 2, 'j': 0, 'k': 0},
            {'value': 0.9977187E-01, 'i': 1, 'j': 1, 'k': 0},
            {'value': 0.7850744E+00, 'i': 2, 'j': 0, 'k': 1},
            {'value': 0.1847187E+00, 'i': 0, 'j': 2, 'k': 2},
            {'value': -0.6893466E-01, 'i': 3, 'j': 0, 'k': 0},
            {'value': 0.9402823E+00, 'i': 0, 'j': 3, 'k': 3},
            {'value': -0.4649396E+00, 'i': 1, 'j': 2, 'k': 2},
            {'value': -0.5417402E+00, 'i': 0, 'j': 4, 'k': 3},
            {'value': 0.1052512E+00, 'i': 3, 'j': 2, 'k': 1},
            {'value': -0.3419544E+00, 'i': 1, 'j': 0, 'k': 3},
            {'value': -0.2585986E+00, 'i': 0, 'j': 4, 'k': 0},
            {'value': 0.3239788E-01, 'i': 6, 'j': 1, 'k': 1},
            {'value': -0.5742804E-01, 'i': 2, 'j': 3, 'k': 0},
            {'value': -0.7892603E+00, 'i': 1, 'j': 1, 'k': 1},
            {'value': -0.5324799E+00, 'i': 0, 'j': 2, 'k': 1},
            {'value': 0.4870383E-02, 'i': 3, 'j': 3, 'k': 0},
            {'value': 0.3483905E+00, 'i': 1, 'j': 4, 'k': 1},
            {'value': 0.3204546E-01, 'i': 4, 'j': 3, 'k': 0},
            {'value': 0.5473935E-02, 'i': 7, 'j': 4, 'k': 3},
            {'value': 0.1084547E-01, 'i': 5, 'j': 0, 'k': 1},
            {'value': -0.1448536E+00, 'i': 4, 'j': 3, 'k': 1},
            {'value': 0.2210349E+00, 'i': 1, 'j': 3, 'k': 0},
            {'value': -0.5244457E-01, 'i': 4, 'j': 1, 'k': 0},
            {'value': 0.3545902E+00, 'i': 0, 'j': 1, 'k': 3},
            {'value': -0.1878683E-01, 'i': 6, 'j': 0, 'k': 2}
        ]

        # 5叶桨KT系数表
        self.kt_coeffs_5 = [
            {'value': 0.5367018E-01, 'i': 0, 'j': 0, 'k': 0},
            {'value': -0.3023566E+00, 'i': 0, 'j': 1, 'k': 0},
            {'value': 0.4333625E+00, 'i': 1, 'j': 0, 'k': 0},
            {'value': -0.1065471E+00, 'i': 0, 'j': 2, 'k': 1},
            {'value': -0.6582904E+00, 'i': 2, 'j': 0, 'k': 3},
            {'value': 0.1189101E+00, 'i': 1, 'j': 3, 'k': 1},
            {'value': -0.4408557E-03, 'i': 6, 'j': 0, 'k': 0},
            {'value': -0.3317857E-01, 'i': 1, 'j': 4, 'k': 1},
            {'value': 0.1151124E+01, 'i': 2, 'j': 0, 'k': 2},
            {'value': 0.1960773E+00, 'i': 0, 'j': 0, 'k': 3},
            {'value': -0.9747062E-01, 'i': 3, 'j': 0, 'k': 1},
            {'value': 0.2036384E+00, 'i': 1, 'j': 1, 'k': 0},
            {'value': -0.2566153E+00, 'i': 1, 'j': 1, 'k': 1},
            {'value': -0.1370242E+00, 'i': 0, 'j': 2, 'k': 0},
            {'value': -0.2874294E+00, 'i': 0, 'j': 0, 'k': 2},
            {'value': -0.2854609E+00, 'i': 2, 'j': 0, 'k': 1}
        ]

        # 5叶桨KQ系数表
        self.kq_coeffs_5 = [
            {'value': -0.9251390E-01, 'i': 0, 'j': 0, 'k': 0},
            {'value': -0.1229000E+00, 'i': 2, 'j': 0, 'k': 0},
            {'value': 0.3050697E+00, 'i': 1, 'j': 1, 'k': 0},
            {'value': -0.2935303E+00, 'i': 0, 'j': 2, 'k': 0},
            {'value': -0.3991474E+00, 'i': 2, 'j': 0, 'k': 1},
            {'value': -0.1022050E+01, 'i': 1, 'j': 1, 'k': 1},
            {'value': 0.1022833E-01, 'i': 7, 'j': 0, 'k': 0},
            {'value': 0.3521100E-02, 'i': 1, 'j': 0, 'k': 3},
            {'value': 0.2552059E-02, 'i': 5, 'j': 2, 'k': 0},
            {'value': 0.2143532E+00, 'i': 0, 'j': 1, 'k': 3},
            {'value': 0.7131110E-03, 'i': 4, 'j': 4, 'k': 0},
            {'value': 0.2078488E+00, 'i': 1, 'j': 2, 'k': 1},
            {'value': 0.6397058E+00, 'i': 1, 'j': 0, 'k': 0},
            {'value': 0.9404846E-03, 'i': 7, 'j': 1, 'k': 0},
            {'value': -0.2930044E-01, 'i': 0, 'j': 1, 'k': 1},
            {'value': -0.7807623E-01, 'i': 0, 'j': 4, 'k': 0},
            {'value': -0.3025523E+00, 'i': 2, 'j': 2, 'k': 3},
            {'value': 0.1855105E+00, 'i': 1, 'j': 3, 'k': 1},
            {'value': -0.6724210E+00, 'i': 2, 'j': 1, 'k': 2},
            {'value': -0.2087142E+00, 'i': 4, 'j': 0, 'k': 3},
            {'value': 0.9400654E+00, 'i': 3, 'j': 0, 'k': 1},
            {'value': 0.9316346E+00, 'i': 2, 'j': 1, 'k': 3},
            {'value': -0.4348397E-01, 'i': 6, 'j': 0, 'k': 0}
        ]

        # 当前选中的系数表
        self.current_kt_coeffs = self.kt_coeffs_4
        self.current_kq_coeffs = self.kq_coeffs_4

    def update_coefficients_by_blade_count(self, blade_count):
        """根据桨叶数更新当前系数表"""
        if blade_count == 4:
            self.current_kt_coeffs = self.kt_coeffs_4
            self.current_kq_coeffs = self.kq_coeffs_4
            return True
        elif blade_count == 5:
            self.current_kt_coeffs = self.kt_coeffs_5
            self.current_kq_coeffs = self.kq_coeffs_5
            return True
        else:
            return False


class StyledButton(QPushButton):
    """自定义样式按钮"""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        # 使用更清晰的字体设置
        self.setFont(QFont("Microsoft YaHei", 9, QFont.Normal))
        self.setMinimumHeight(28)  # 稍微增加高度以改善显示
        self.setStyleSheet("""
            QPushButton {
                background-color: #2c3e50;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px 10px;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:pressed {
                background-color: #1a2530;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
                color: #7f8c8d;
            }
        """)


class StyledGroupBox(QGroupBox):
    """自定义样式分组框"""

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                margin-top: 0.5ex;
                padding-top: 6px;
                background-color: #f8f9fa;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
                font-weight: 500;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 3px 0 3px;
                color: #2c3e50;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
                font-weight: 600;
            }
        """)


class StyledTableWidget(QTableWidget):
    """自定义样式表格控件"""

    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns, parent)
        self.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f8f9fa;
                gridline-color: #dee2e6;
                border: 1px solid #dee2e6;
                border-radius: 3px;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
                font-weight: normal;
                gridline-color: #d0d0d0;
            }
            QTableWidget::item {
                padding: 3px;
                border-bottom: 1px solid #dee2e6;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background-color: #2c3e50;
                color: white;
                padding: 4px;
                border: none;
                font-weight: 600;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
            }
        """)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setStretchLastSection(True)


class StyledLineEdit(QLineEdit):
    """自定义样式输入框"""

    def __init__(self, placeholder="", parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        # 使用更清晰的字体
        self.setFont(QFont("Microsoft YaHei", 9))
        self.setStyleSheet("""
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 2px;
                padding: 4px;
                background-color: white;
                font-family: "Microsoft YaHei", "Times New Roman";
                font-size: 9pt;
                font-weight: normal;
                selection-background-color: #3498db;
            }
            QLineEdit:focus {
                border-color: #3498db;
                background-color: #f8f9fa;
            }
            QLineEdit:disabled {
                background-color: #ecf0f1;
                color: #7f8c8d;
            }
        """)


class StyledTextEdit(QTextEdit):
    """自定义样式文本编辑框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置清晰字体
        self.setFont(QFont("Microsoft YaHei", 9))
        self.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 2px;
                padding: 5px;
                font-family: "Microsoft YaHei", "Times New Roman";
                font-size: 9pt;
                font-weight: normal;
                line-height: 1.2;
            }
            QTextEdit:focus {
                border-color: #3498db;
            }
        """)


class PropellerDesignSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置高DPI支持以改善字体渲染
        self.setup_high_dpi_support()

        self.setWindowTitle("船舶螺旋桨图谱设计系统")
        self.setGeometry(50, 50, 1000, 600)

        # 设置应用样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QLabel {
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
                font-weight: normal;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
                margin: 2px;
            }
            QTabBar::tab {
                background-color: #95a5a6;
                color: white;
                padding: 8px 15px;
                margin: 1px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                font-weight: 600;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
            }
            QTabBar::tab:selected {
                background-color: #2c3e50;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #7f8c8d;
            }
        """)

        # 初始化变量
        self.res = {}
        self.opt_res = {}
        self.mass_details = []
        self.au_coeffs = AUCoefficients()
        self.cavitation_results = {}
        self.optimum_results = {}
        self.blade_count = 4

        # 创建主界面
        self.init_ui()

    def setup_high_dpi_support(self):
        """设置高DPI支持以改善字体渲染"""
        try:
            # 启用高DPI缩放
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        except:
            pass  # 如果系统不支持高DPI，则忽略

    def init_ui(self):
        # 创建中央控件
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            background-color: #ecf0f1;
            font-family: "Microsoft YaHei", "SimSun";
            font-size: 9pt;
        """)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建头部
        header = QWidget()
        header.setFixedHeight(45)
        header.setStyleSheet("""
            background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #2c3e50, stop: 1 #34495e);
            color: white;
            font-weight: bold;
            border-bottom: 1px solid #2980b9;
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 0, 15, 0)

        title = QLabel("船舶螺旋桨图谱设计系统")
        title.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            color: white; 
            font-family: "Microsoft YaHei", "SimSun";
            font-weight: 600;
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        version = QLabel("v1.0")
        version.setStyleSheet("""
            font-size: 10px; 
            color: #bdc3c7; 
            font-family: "Microsoft YaHei", "SimSun";
        """)
        header_layout.addWidget(version)

        main_layout.addWidget(header)

        # 创建标签页控件
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                background-color: white;
                margin: 2px;
            }
            QTabBar::tab {
                background-color: #95a5a6;
                color: white;
                padding: 8px 15px;
                margin: 1px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                font-weight: 600;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
            }
            QTabBar::tab:selected {
                background-color: #2c3e50;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #7f8c8d;
            }
        """)

        # 添加标签页
        self.tabs.addTab(self.create_max_speed_tab(), "🚀 最大航速")
        self.tabs.addTab(self.create_optimum_selection_tab(), "🎯 最佳要素")
        self.tabs.addTab(self.create_strength_tab(), "🛡️ 强度校核")
        self.tabs.addTab(self.create_pitch_correction_tab(), "📏 螺距修正")
        self.tabs.addTab(self.create_mass_inertia_tab(), "⚖️ 质量惯性")
        self.tabs.addTab(self.create_open_water_tab(), "🌊 敞水曲线")
        self.tabs.addTab(self.create_mooring_tab(), "⚓ 系柱计算")
        self.tabs.addTab(self.create_voyage_characteristics_tab(), "📊 航行特性")

        main_layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)

    def create_styled_input(self, label_text, default_value=""):
        """创建带标签的样式化输入"""
        label = QLabel(label_text)
        label.setStyleSheet("""
            font-weight: 600; 
            color: #2c3e50; 
            font-family: "Microsoft YaHei", "SimSun"; 
            font-size: 9pt;
        """)
        line_edit = StyledLineEdit(default_value)
        return label, line_edit

    # ===================== 1. 最大航速 =====================
    def create_max_speed_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)  # 增加间距
        lay.setContentsMargins(8, 8, 8, 8)

        # 输入参数组
        input_group = StyledGroupBox("设计参数 (已考虑10%功率储备)")
        form_layout = QFormLayout()
        form_layout.setSpacing(6)  # 增加间距
        form_layout.setLabelAlignment(Qt.AlignRight)

        # 创建样式化输入
        self.ps_label, self.ps_input = self.create_styled_input("主机功率 Ps (kW)", "6222")
        self.n_label, self.n_input = self.create_styled_input("主机转速 N (r/min)", "155")
        self.etas_label, self.etas_input = self.create_styled_input("轴系效率 ηs", "0.97")
        self.etar_label, self.etar_input = self.create_styled_input("相对旋转效率 ηR", "1")
        self.w_label, self.w_input = self.create_styled_input("伴流分数 w", "0.35")
        self.t_label, self.t_input = self.create_styled_input("推力减额分数 t", "0.21")
        self.vs_label, self.vs_input = self.create_styled_input("设计航速 Vs (kn)", "15")

        # 添加桨叶数选择
        blade_layout = QHBoxLayout()
        blade_label = QLabel("桨叶数:")
        blade_label.setStyleSheet("""
            font-weight: 600; 
            color: #2c3e50; 
            font-family: "Microsoft YaHei", "SimSun"; 
            font-size: 9pt;
        """)
        blade_layout.addWidget(blade_label)
        self.blade_combo = QComboBox()
        self.blade_combo.addItems(["4", "5"])
        self.blade_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 2px;
                padding: 4px;
                background-color: white;
                font-family: "Microsoft YaHei", "Times New Roman";
                font-size: 9pt;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #bdc3c7;
                border-left-style: solid;
                border-top-right-radius: 2px;
                border-bottom-right-radius: 2px;
            }
        """)
        self.blade_combo.currentIndexChanged.connect(self.on_blade_count_changed)
        blade_layout.addWidget(self.blade_combo)
        blade_layout.addStretch()
        form_layout.addRow(blade_layout)

        # 添加所有输入到表单
        form_layout.addRow(self.ps_label, self.ps_input)
        form_layout.addRow(self.n_label, self.n_input)
        form_layout.addRow(self.etas_label, self.etas_input)
        form_layout.addRow(self.etar_label, self.etar_input)
        form_layout.addRow(self.w_label, self.w_input)
        form_layout.addRow(self.t_label, self.t_input)
        form_layout.addRow(self.vs_label, self.vs_input)

        input_group.setLayout(form_layout)
        lay.addWidget(input_group)

        # PE曲线输入
        pe_group = StyledGroupBox("有效功率曲线 (格式: 航速,...;功率,...)")
        pe_layout = QVBoxLayout()
        self.pe_edit = StyledLineEdit("12,13,14,15,16,17;1497,1953,2505,3213,4070,5161")
        pe_layout.addWidget(self.pe_edit)
        pe_group.setLayout(pe_layout)
        lay.addWidget(pe_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_calc_speed = StyledButton("计算航速")
        self.btn_clear = StyledButton("清空数据")
        self.btn_plot_speed = StyledButton("绘制曲线")

        btn_layout.addWidget(self.btn_calc_speed)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_plot_speed)
        lay.addLayout(btn_layout)

        # 结果表格
        table_group = StyledGroupBox("计算结果")
        table_layout = QVBoxLayout()
        self.tbl_speed = StyledTableWidget(3, 6)
        self.tbl_speed.setHorizontalHeaderLabels(["型号", "Vmax (kn)", "P/D", "δ", "D (m)", "η₀"])
        # 设置初始桨叶类型
        self.tbl_speed.setVerticalHeaderLabels(["MAU4-40", "MAU4-55", "MAU4-70"])
        table_layout.addWidget(self.tbl_speed)
        table_group.setLayout(table_layout)
        lay.addWidget(table_group)

        # 连接信号
        self.btn_calc_speed.clicked.connect(self.calculate_max_speed)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_plot_speed.clicked.connect(self.plot_max_speed_results)

        return w

    def on_blade_count_changed(self):
        """当桨叶数改变时更新界面"""
        self.blade_count = int(self.blade_combo.currentText())
        # 更新表格的行标签
        if self.blade_count == 4:
            self.tbl_speed.setVerticalHeaderLabels(["MAU4-40", "MAU4-55", "MAU4-70"])
        else:  # 5叶桨
            self.tbl_speed.setVerticalHeaderLabels(["MAU5-50", "MAU5-65", "MAU5-80"])

    def calculate_max_speed(self):
        try:
            # 首先检查必要的输入控件是否存在
            required_inputs = ['ps_input', 'n_input', 'etas_input', 'etar_input', 'w_input', 't_input', 'pe_edit']
            for input_name in required_inputs:
                if not hasattr(self, input_name):
                    QMessageBox.critical(self, "界面错误", f"界面组件 {input_name} 未正确初始化，请重启程序")
                    return

            # 获取基本参数 - 添加默认值处理
            ps_text = self.ps_input.text().strip() or "6222"
            n_text = self.n_input.text().strip() or "155"
            eta_s_text = self.etas_input.text().strip() or "0.97"
            eta_r_text = self.etar_input.text().strip() or "1"
            w_text = self.w_input.text().strip() or "0.35"
            t_text = self.t_input.text().strip() or "0.21"

            ps = float(ps_text)
            n = float(n_text)
            eta_s = float(eta_s_text)
            eta_r = float(eta_r_text)
            w = float(w_text)
            t = float(t_text)

            # 解析有效功率曲线数据
            pe_text = self.pe_edit.text().strip()
            if not pe_text:
                # 使用默认值
                pe_text = "12,13,14,15,16,17;1497,1953,2505,3213,4070,5161"

            if ';' not in pe_text:
                raise ValueError("有效功率曲线格式错误，应使用分号分隔航速和功率")

            p = pe_text.split(';')
            if len(p) != 2:
                raise ValueError("有效功率曲线格式错误，应包含航速和功率两部分")

            # 清理数据并转换为浮点数
            speeds_str = p[0].split(',')
            pes_str = p[1].split(',')

            # 移除可能的空字符串
            speeds_str = [s.strip() for s in speeds_str if s.strip()]
            pes_str = [p.strip() for p in pes_str if p.strip()]

            if len(speeds_str) == 0 or len(pes_str) == 0:
                raise ValueError("航速或功率数据不能为空")

            if len(speeds_str) != len(pes_str):
                raise ValueError(f"航速和功率数量不一致: {len(speeds_str)}个航速 vs {len(pes_str)}个功率")

            # 转换为浮点数
            speeds = []
            pes = []
            for s in speeds_str:
                try:
                    speeds.append(float(s))
                except ValueError:
                    raise ValueError(f"无效的航速值: '{s}'")

            for p_val in pes_str:
                try:
                    pes.append(float(p_val))
                except ValueError:
                    raise ValueError(f"无效的功率值: '{p_val}'")

            # 验证数据范围
            if min(speeds) <= 0:
                raise ValueError("航速必须大于0")
            if min(pes) <= 0:
                raise ValueError("功率必须大于0")

            # 计算推进功率 - 注意：这里考虑了10%功率储备
            pd = ps * 0.9 * eta_s * eta_r
            eta_h = (1 - t) / (1 - w)

            self.res = {
                'PD': pd, 'N': n, 'w': w, 't': t,
                'eta_H': eta_h, 'speeds': speeds, 'pes': pes,
                'Ps': ps, 'eta_s': eta_s, 'eta_r': eta_r
            }

            # 根据桨叶数选择型号
            if self.blade_count == 4:
                types = ["MAU4-40", "MAU4-55", "MAU4-70"]
            else:  # 5叶桨
                types = ["MAU5-50", "MAU5-65", "MAU5-80"]

            print(f"计算参数: PD={pd:.1f}kW, N={n}rpm, w={w:.3f}, t={t:.3f}")
            print(f"航速范围: {min(speeds)}-{max(speeds)}kn, 功率范围: {min(pes)}-{max(pes)}kW")

            # 计算每个型号的结果
            for row, tp in enumerate(types):
                try:
                    vmax, p_d, delta, D, eta0 = self.calculate_for_type(tp, speeds, pes)
                    print(f"型号 {tp}: Vmax={vmax:.2f}kn, P/D={p_d:.3f}, δ={delta:.1f}, D={D:.3f}m, η0={eta0:.4f}")

                    # 更新表格
                    for col, val in enumerate(
                            [tp, f"{vmax:.2f}", f"{p_d:.3f}", f"{delta:.3f}", f"{D:.3f}", f"{eta0:.4f}"]):
                        item = QTableWidgetItem(val)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.tbl_speed.setItem(row, col, item)

                except Exception as e:
                    print(f"计算型号 {tp} 时出错: {str(e)}")
                    # 在表格中显示错误信息
                    for col in range(6):
                        error_msg = "计算错误" if col == 0 else ""
                        item = QTableWidgetItem(error_msg)
                        item.setTextAlignment(Qt.AlignCenter)
                        self.tbl_speed.setItem(row, col, item)

            QMessageBox.information(self, "成功", "最大航速计算完成")

        except ValueError as e:
            QMessageBox.critical(self, "输入错误", f"参数格式错误: {str(e)}\n\n请检查所有输入框是否填写了有效的数字。")
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算过程中发生错误: {str(e)}")

    def calculate_for_type(self, tp, speeds, pes):
        bp = self.get_bp_data(tp)

        # 使用更高精度的插值方法 - 三次样条插值
        interp_delta = CubicSpline(bp['sqrt'], bp['delta'])
        interp_pd = CubicSpline(bp['sqrt'], bp['p_d'])
        interp_eta = CubicSpline(bp['sqrt'], bp['eta'])

        try:
            # 使用三次样条插值拟合PE曲线
            pe_func = CubicSpline(speeds, pes)
        except:
            # 如果三次样条失败，使用Akima插值
            pe_func = Akima1DInterpolator(speeds, pes)

        def pte(V):
            VA = (1 - self.res['w']) * V
            Bp = (self.res['N'] * np.sqrt(self.res['PD'])) / (VA ** 2.5) * 1.166
            sqrt_bp = np.sqrt(Bp)
            eta0_val = float(interp_eta(sqrt_bp))
            return self.res['PD'] * self.res['eta_H'] * eta0_val

        vmax = max(speeds)
        try:
            # 使用更精确的求解方法
            vmax = float(fsolve(lambda v: pte(v) - pe_func(v), vmax, xtol=1e-6)[0])
            vmax = max(min(speeds), min(max(speeds), vmax))  # 限制在有效范围内
        except:
            vmax = max(speeds)

        VA = (1 - self.res['w']) * vmax
        Bp = (self.res['N'] * np.sqrt(self.res['PD'])) / (VA ** 2.5) * 1.166
        sqrt_bp = np.sqrt(Bp)
        delta = float(interp_delta(sqrt_bp))
        p_d = float(interp_pd(sqrt_bp))
        eta0 = float(interp_eta(sqrt_bp))
        D = (delta * VA) / self.res['N']
        return vmax, p_d, delta, D, eta0

    def get_bp_data(self, tp):
        # 使用新的MAU4系列图谱数据
        if tp == "MAU4-40":
            # 过滤掉无效数据点
            sqrt_vals = [2.43, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5,
                         6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.07]
            delta_vals = [32.1337, 33.0527, 35.6638, 38.8661, 41.6769, 44.1932, 47.0351, 49.4805, 52.5412, 55.0486,
                          57.6749, 60.6623, 62.792, 65.4302, 68, 70.9688, 73.4625, 75.5852, 78.1068, 80.4074, 82.4419,
                          84.9337, 87.5578, 89.5028, 92.4124, 94, 96, 98.9679, 100.738, 102.8976, 105.1432, 107.2132,
                          107.7646]
            p_d_vals = [1.11168, 1.08883, 1.02299, 0.95694, 0.91488, 0.87439, 0.83645, 0.81378, 0.78523, 0.75768,
                        0.7395, 0.72046, 0.7014, 0.68219, 0.672, 0.65867, 0.64867, 0.6405, 0.63474, 0.6282, 0.61852,
                        0.61276, 0.60609, 0.60127, 0.59407, 0.59, 0.582, 0.57692, 0.57457, 0.57087, 0.56892, 0.56446,
                        0.56274]
            eta_vals = [0.76169, 0.75949, 0.75125, 0.73741, 0.72345, 0.70847, 0.69006, 0.67778, 0.66364, 0.65216,
                        0.64142, 0.62654, 0.61688, 0.60503, 0.592, 0.5806, 0.56823, 0.55987, 0.5475, 0.53862, 0.53122,
                        0.52179, 0.51156, 0.50677, 0.49276, 0.485, 0.48, 0.47189, 0.4657, 0.45899, 0.45096, 0.44431,
                        0.44269]

            return {
                'sqrt': sqrt_vals,
                'delta': delta_vals,
                'p_d': p_d_vals,
                'eta': eta_vals
            }
        elif tp == "MAU4-55":
            # 更新为新的MAU4-55数据
            sqrt_vals = [4.586, 4.971, 5.419, 5.945, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 13.01]
            delta_vals = [55.6, 58.8, 63.4, 69.1, 74.0, 78.5, 82.8, 86.9, 90.8, 94.5, 98.0, 101.0, 107.0, 132.3]
            p_d_vals = [0.807, 0.774, 0.742, 0.711, 0.680, 0.650, 0.620, 0.595, 0.570, 0.545, 0.525, 0.505, 0.470,
                        0.400]
            eta_vals = [0.634, 0.614, 0.592, 0.565, 0.540, 0.515, 0.490, 0.465, 0.440, 0.415, 0.390, 0.365, 0.330,
                        0.260]

            return {
                'sqrt': sqrt_vals,
                'delta': delta_vals,
                'p_d': p_d_vals,
                'eta': eta_vals
            }
        elif tp == "MAU4-70":
            # 过滤掉0值数据点
            sqrt_vals = [2.65, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7,
                         7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 10.07]
            delta_vals = [32, 33.3173, 36.5182, 39.3473, 42, 45.3888, 48, 51.0038, 53.4893, 56, 58, 60.6577, 63.5746,
                          65.5697, 68, 70.4972, 73.0047, 75.485, 78, 80, 83.0763, 85.5071, 87.658, 89.4571, 92.5176,
                          94.4395, 97.0423, 99.2439, 101.1474, 103.479, 106, 106.5729]
            p_d_vals = [1.21, 1.17707, 1.09708, 1.02612, 0.97, 0.91193, 0.88, 0.84889, 0.83298, 0.81, 0.79, 0.76611,
                        0.75225, 0.74212, 0.73, 0.71751, 0.70499, 0.69253, 0.68, 0.67, 0.66274, 0.65275, 0.64117,
                        0.63713, 0.62315, 0.62297, 0.62177, 0.61714, 0.61053, 0.60743, 0.606, 0.6045]
            eta_vals = [0.705, 0.69778, 0.68784, 0.6728, 0.66, 0.64316, 0.63, 0.61942, 0.60725, 0.595, 0.585, 0.57443,
                        0.56195, 0.55306, 0.541, 0.531, 0.5215, 0.512, 0.503, 0.495, 0.48542, 0.47661, 0.47013, 0.46359,
                        0.45545, 0.44874, 0.44245, 0.43662, 0.42907, 0.42284, 0.416, 0.41471]

            return {
                'sqrt': sqrt_vals,
                'delta': delta_vals,
                'p_d': p_d_vals,
                'eta': eta_vals
            }
        # 5叶桨数据保持不变
        elif tp == "MAU5-50":
            return {'sqrt': [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                    'delta': [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0],
                    'p_d': [1.10, 1.05, 0.95, 0.88, 0.82, 0.78, 0.74, 0.71, 0.68, 0.65, 0.63, 0.61],
                    'eta': [0.75, 0.73, 0.70, 0.67, 0.64, 0.61, 0.58, 0.55, 0.52, 0.49, 0.46, 0.43]}
        elif tp == "MAU5-65":
            return {'sqrt': [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                    'delta': [32.0, 38.0, 44.0, 50.0, 56.0, 62.0, 68.0, 74.0, 80.0, 86.0, 92.0, 98.0],
                    'p_d': [1.08, 1.02, 0.93, 0.85, 0.79, 0.74, 0.70, 0.67, 0.64, 0.61, 0.59, 0.57],
                    'eta': [0.72, 0.70, 0.67, 0.64, 0.61, 0.58, 0.55, 0.52, 0.49, 0.46, 0.43, 0.40]}
        elif tp == "MAU5-80":
            return {'sqrt': [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
                    'delta': [34.0, 41.0, 48.0, 55.0, 62.0, 69.0, 76.0, 83.0, 90.0, 97.0, 104.0, 111.0],
                    'p_d': [1.05, 0.98, 0.90, 0.83, 0.77, 0.72, 0.68, 0.65, 0.62, 0.59, 0.57, 0.55],
                    'eta': [0.68, 0.66, 0.63, 0.60, 0.57, 0.54, 0.51, 0.48, 0.45, 0.42, 0.39, 0.36]}
        else:
            # 默认返回MAU4-55的数据
            return {'sqrt': [4.586, 4.971, 5.419, 5.945, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 13.01],
                    'delta': [55.6, 58.8, 63.4, 69.1, 74.0, 78.5, 82.8, 86.9, 90.8, 94.5, 98.0, 101.0, 107.0, 132.3],
                    'p_d': [0.807, 0.774, 0.742, 0.711, 0.680, 0.650, 0.620, 0.595, 0.570, 0.545, 0.525, 0.505, 0.470,
                            0.400],
                    'eta': [0.634, 0.614, 0.592, 0.565, 0.540, 0.515, 0.490, 0.465, 0.440, 0.415, 0.390, 0.365, 0.330,
                            0.260]}

    def plot_max_speed_results(self):
        """绘制最大航速计算结果曲线 - 改进版本"""
        if not self.res:
            QMessageBox.warning(self, "警告", "请先完成最大航速计算")
            return

        try:
            # 获取有效功率曲线数据
            pe_text = self.pe_edit.text().strip()
            if not pe_text:
                pe_text = "12,13,14,15,16,17;1497,1953,2505,3213,4070,5161"

            p = pe_text.split(';')
            if len(p) != 2:
                raise ValueError("有效功率曲线格式错误")

            # 清理数据，移除可能的空值
            speeds_str = [s.strip() for s in p[0].split(',') if s.strip()]
            pes_str = [p_val.strip() for p_val in p[1].split(',') if p_val.strip()]

            if not speeds_str or not pes_str:
                raise ValueError("航速或功率数据为空")

            speeds = list(map(float, speeds_str))
            pes = list(map(float, pes_str))

            # 创建绘图窗口 - 调整大小为800x1000
            self.plot_window = QDialog(self)
            self.plot_window.setWindowTitle("最大航速计算结果")
            self.plot_window.setGeometry(150, 150, 800, 1000)  # 调整大小

            # 创建图表
            fig = Figure(figsize=(8, 10), dpi=100)  # 调整图形大小
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self.plot_window)

            # 设置全局字体
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'SimHei',
                'axes.unicode_minus': False
            })

            # 根据桨叶数确定型号
            if self.blade_count == 4:
                types = ["MAU4-40", "MAU4-55", "MAU4-70"]
                colors = ['red', 'blue', 'green']
                line_styles = ['-', '--', '-.']  # 不同线型
                markers = ['o', 's', '^']  # 不同标记
                labels = ['MAU4-40', 'MAU4-55', 'MAU4-70']
            else:  # 5叶桨
                types = ["MAU5-50", "MAU5-65", "MAU5-80"]
                colors = ['red', 'blue', 'green']
                line_styles = ['-', '--', '-.']
                markers = ['o', 's', '^']
                labels = ['MAU5-50', 'MAU5-65', 'MAU5-80']

            # 生成航速范围 - 增加采样点以提高光滑度
            v_min = min(speeds)
            v_max = max(speeds)
            v_range = np.linspace(v_min, v_max, 200)  # 增加采样点

            # 创建四个子图
            ax1 = fig.add_subplot(4, 1, 1)  # η0
            ax2 = fig.add_subplot(4, 1, 2)  # P/D
            ax3 = fig.add_subplot(4, 1, 3)  # δ
            ax4 = fig.add_subplot(4, 1, 4)  # PE和PTE

            # 设置子图标题和标签
            ax1.set_ylabel('敞水效率 η₀', fontsize=12)
            ax1.grid(True, alpha=0.3)

            ax2.set_ylabel('螺距比 P/D', fontsize=12)
            ax2.grid(True, alpha=0.3)

            ax3.set_ylabel('直径系数 δ', fontsize=12)
            ax3.grid(True, alpha=0.3)

            ax4.set_ylabel('功率 PE, PTE (kW)', fontsize=12)
            ax4.set_xlabel('航速 V (kn)', fontsize=12)
            ax4.grid(True, alpha=0.3)

            # 对每个型号计算曲线
            intersection_points = []  # 存储交点信息

            for i, tp in enumerate(types):
                # 获取该型号的Bp数据
                bp_data = self.get_bp_data(tp)

                # 创建更高精度的插值函数 - 使用三次样条插值
                interp_delta = CubicSpline(bp_data['sqrt'], bp_data['delta'])
                interp_pd = CubicSpline(bp_data['sqrt'], bp_data['p_d'])
                interp_eta = CubicSpline(bp_data['sqrt'], bp_data['eta'])

                # 计算每个航速下的参数
                p_d_vals = []
                delta_vals = []
                eta0_vals = []
                pte_vals = []

                for v in v_range:
                    try:
                        VA = (1 - self.res['w']) * v
                        Bp = (self.res['N'] * np.sqrt(self.res['PD'])) / (VA ** 2.5) * 1.166
                        sqrt_bp = np.sqrt(Bp)

                        delta_val = float(interp_delta(sqrt_bp))
                        p_d_val = float(interp_pd(sqrt_bp))
                        eta0_val = float(interp_eta(sqrt_bp))
                        pte_val = self.res['PD'] * self.res['eta_H'] * eta0_val

                        p_d_vals.append(p_d_val)
                        delta_vals.append(delta_val)
                        eta0_vals.append(eta0_val)
                        pte_vals.append(pte_val)
                    except:
                        p_d_vals.append(0)
                        delta_vals.append(0)
                        eta0_vals.append(0)
                        pte_vals.append(0)

                # 绘制曲线 - 使用不同颜色和线型
                ax1.plot(v_range, eta0_vals, color=colors[i], linestyle=line_styles[i],
                         linewidth=2, label=labels[i], marker=markers[i], markersize=4, markevery=20)
                ax2.plot(v_range, p_d_vals, color=colors[i], linestyle=line_styles[i],
                         linewidth=2, label=labels[i], marker=markers[i], markersize=4, markevery=20)
                ax3.plot(v_range, delta_vals, color=colors[i], linestyle=line_styles[i],
                         linewidth=2, label=labels[i], marker=markers[i], markersize=4, markevery=20)
                ax4.plot(v_range, pte_vals, color=colors[i], linestyle=line_styles[i],
                         linewidth=2, label=f'{labels[i]} PTE', marker=markers[i], markersize=4, markevery=20)

                # 计算PE和PTE的交点
                try:
                    # 使用三次样条插值拟合PE曲线
                    pe_func = CubicSpline(speeds, pes)
                    pe_vals = pe_func(v_range)

                    # 找到交点
                    for j in range(len(v_range) - 1):
                        if (pte_vals[j] - pe_vals[j]) * (pte_vals[j + 1] - pe_vals[j + 1]) <= 0:
                            # 线性插值求交点
                            t = (pe_vals[j] - pte_vals[j]) / (
                                    pte_vals[j + 1] - pte_vals[j] - (pe_vals[j + 1] - pe_vals[j]))
                            v_intersect = v_range[j] + t * (v_range[j + 1] - v_range[j])
                            p_intersect = pe_vals[j] + t * (pe_vals[j + 1] - pe_vals[j])

                            intersection_points.append((v_intersect, p_intersect, labels[i]))

                            # 在所有子图中绘制竖直虚线
                            for ax in [ax1, ax2, ax3, ax4]:
                                ax.axvline(x=v_intersect, color=colors[i], linestyle=':', alpha=0.7, linewidth=2)

                            # 在PTE子图中标记交点
                            ax4.plot(v_intersect, p_intersect, 'o', color=colors[i], markersize=8)
                            ax4.annotate(f'{v_intersect:.2f} kn',
                                         xy=(v_intersect, p_intersect),
                                         xytext=(10, 10), textcoords='offset points',
                                         fontsize=9, color=colors[i])
                            break
                except Exception as e:
                    print(f"计算交点时出错: {str(e)}")

            # 绘制有效功率曲线
            try:
                pe_func = CubicSpline(speeds, pes)
                pe_vals = pe_func(v_range)
                ax4.plot(v_range, pe_vals, 'k-', linewidth=3, label='有效功率 PE')
            except Exception as e:
                print(f"绘制PE曲线时出错: {str(e)}")

            # 添加图例
            ax1.legend(loc='best', fontsize=10)
            ax2.legend(loc='best', fontsize=10)
            ax3.legend(loc='best', fontsize=10)
            ax4.legend(loc='best', fontsize=10)

            # 设置标题
            fig.suptitle('螺旋桨性能参数随航速变化曲线',
                         fontsize=14, fontweight='bold')

            # 调整子图间距
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            # 添加到布局
            layout = QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            self.plot_window.setLayout(layout)
            self.plot_window.show()

        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘制曲线时发生错误: {str(e)}")

    # ===================== 2. 最佳要素确定 =====================
    def create_optimum_selection_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 空泡校核参数输入
        input_group = StyledGroupBox("空泡校核参数")
        input_layout = QFormLayout()
        input_layout.setSpacing(6)
        input_layout.setLabelAlignment(Qt.AlignRight)

        self.depth_label, self.depth_input = self.create_styled_input("桨轴沉深 hs (m)", "5.0")
        self.temp_label, self.temp_input = self.create_styled_input("计算温度 t (°C)", "15")
        self.pv_label, self.pv_input = self.create_styled_input("饱和蒸汽压 Pv (Pa)", "1706")
        self.p0_label, self.p0_input = self.create_styled_input("大气压力 P0 (Pa)", "101325")

        input_layout.addRow(self.depth_label, self.depth_input)
        input_layout.addRow(self.temp_label, self.temp_input)
        input_layout.addRow(self.pv_label, self.pv_input)
        input_layout.addRow(self.p0_label, self.p0_input)
        input_group.setLayout(input_layout)

        # τc 计算方式
        self.cav_combo = StyledGroupBox("τc 计算方式")
        cav_combo_layout = QHBoxLayout()
        self.rb_wag = QRadioButton("瓦格宁根水池限界线")
        self.rb_ber = QRadioButton("柏利尔商船限界线")
        self.rb_wag.setChecked(True)

        # 样式化单选按钮
        radio_style = """
            QRadioButton {
                font-weight: 600;
                color: #2c3e50;
                padding: 4px;
                font-family: "Microsoft YaHei", "SimSun";
                font-size: 9pt;
            }
            QRadioButton::indicator {
                width: 12px;
                height: 12px;
            }
        """
        self.rb_wag.setStyleSheet(radio_style)
        self.rb_ber.setStyleSheet(radio_style)

        cav_combo_layout.addWidget(self.rb_wag)
        cav_combo_layout.addWidget(self.rb_ber)
        self.cav_combo.setLayout(cav_combo_layout)

        button_layout = QHBoxLayout()
        self.calculate_cav_btn = StyledButton("空泡校核")
        button_layout.addWidget(self.calculate_cav_btn)

        # 空泡校核结果表格
        table_group = StyledGroupBox("空泡校核结果")
        table_layout = QVBoxLayout()
        self.cavitation_table = StyledTableWidget(9, 4)
        self.cavitation_table.setHorizontalHeaderLabels(["计算公式", "MAU4-40", "MAU4-55", "MAU4-70"])
        self.cavitation_table.setEditTriggers(QTableWidget.NoEditTriggers)

        table_items = [
            ("PD", "", "", ""),
            ("Vmax", "", "", ""),
            ("VA = 0.5144 × Vmax × (1 - w)", "", "", ""),
            ("ω = 0.7πND/60", "", "", ""),
            ("V₀.₇ᴿ² = VA² + ω²", "", "", ""),
            ("σ = (P₀ + ρghₛ - Pᵥ) / (0.5ρV₀.₇ᴿ²)", "", "", ""),
            ("τc = f(σ)", "", "", ""),
            ("T = PD × η₀ × 1000 / VA", "", "", ""),
            ("Aᴇ/A₀ = T / (0.5ρV₀.₇ᴿ²τc πD²/4 (1.067-0.229 P/D))", "", "", "")
        ]

        for row, items in enumerate(table_items):
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft if col == 0 else Qt.AlignCenter)
                self.cavitation_table.setItem(row, col, item)

        self.cavitation_table.setColumnWidth(0, 300)
        for i in range(1, 4):
            self.cavitation_table.setColumnWidth(i, 90)

        table_layout.addWidget(self.cavitation_table)
        table_group.setLayout(table_layout)

        # 最佳要素确定部分
        description_group = StyledGroupBox("最佳要素确定说明")
        description_layout = QVBoxLayout()
        description = QLabel("此功能根据空泡校核结果确定最佳螺旋桨要素：\n"
                             "1. 绘制AE/A₀、P/D、D、η₀、Vmax 随盘面比变化光滑曲线\n"
                             "2. 绘制从(0.4,0)到(0.7, 上框最高点)的对角线（跨全图）\n"
                             "3. 找到交点并确定满足空泡要求的最佳要素")
        description.setFont(QFont("Microsoft YaHei", 9))
        description.setStyleSheet("""
            background-color: #f8f9fa; 
            padding: 6px; 
            border-radius: 3px; 
            color: #2c3e50; 
            font-family: "Microsoft YaHei", "SimSun"; 
            font-size: 9pt;
            line-height: 1.4;
        """)
        description_layout.addWidget(description)
        description_group.setLayout(description_layout)

        btn_layout2 = QHBoxLayout()
        self.plot_btn = StyledButton("确定最佳要素")
        self.plot_btn.setEnabled(False)
        self.results_btn = StyledButton("显示结果")
        self.results_btn.setEnabled(False)
        btn_layout2.addWidget(self.plot_btn)
        btn_layout2.addWidget(self.results_btn)
        btn_layout2.addStretch()

        result_group = StyledGroupBox("最佳要素结果")
        result_layout = QVBoxLayout()
        self.result_text = StyledTextEdit()
        self.result_text.setFont(QFont("Microsoft YaHei", 9))
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)

        # 添加到布局
        layout.addWidget(input_group)
        layout.addWidget(self.cav_combo)
        layout.addLayout(button_layout)
        layout.addWidget(table_group)
        layout.addWidget(description_group)
        layout.addLayout(btn_layout2)
        layout.addWidget(result_group)

        self.calculate_cav_btn.clicked.connect(self.calculate_cavitation)
        self.plot_btn.clicked.connect(self.plot_curves_and_find_optimum)
        self.results_btn.clicked.connect(self.show_optimum_results)

        return w

    def get_tau_c(self, sigma, source='wag'):
        """统一 τc 计算"""
        if source == 'wag':
            try:
                tau_c = float(Akima1DInterpolator(SIGMA_WAG, TAU_C_WAG)(sigma))
            except:
                tau_c = 0.15
        else:  # ber
            if sigma < 0.36:
                tau_c = 0.14
            elif sigma > 1.82:
                tau_c = 0.35
            else:
                try:
                    tau_c = float(Akima1DInterpolator(SIGMA_BER, TAU_C_BER)(sigma))
                except:
                    tau_c = 0.15
        return max(0.05, min(0.5, tau_c))

    def calculate_cavitation(self):
        try:
            if not self.res:
                QMessageBox.warning(self, "警告", "请先完成最大航速计算")
                return

            # 检查最大航速计算结果表格是否有数据
            if self.tbl_speed.rowCount() == 0 or self.tbl_speed.item(0, 1) is None:
                QMessageBox.warning(self, "警告", "最大航速计算结果为空，请先完成最大航速计算")
                return

            hs_text = self.depth_input.text().strip() or "5.0"
            t_text = self.temp_input.text().strip() or "15"
            pv_text = self.pv_input.text().strip() or "1706"
            p0_text = self.p0_input.text().strip() or "101325"

            hs = float(hs_text)
            t = float(t_text)
            pv = float(pv_text)
            p0 = float(p0_text)

            rho = 1025.0
            g = 9.81
            p0_total = p0 + rho * g * hs

            self.cavitation_results = {}

            # 根据桨叶数确定型号
            if self.blade_count == 4:
                propeller_types = ["MAU4-40", "MAU4-55", "MAU4-70"]
            else:  # 5叶桨
                propeller_types = ["MAU5-50", "MAU5-65", "MAU5-80"]

            # 更新表格列标题
            self.cavitation_table.setColumnCount(len(propeller_types) + 1)
            headers = ["计算公式"] + propeller_types
            self.cavitation_table.setHorizontalHeaderLabels(headers)

            for col, propeller_type in enumerate(propeller_types, start=1):
                # 从最大航速计算结果获取数据
                row = propeller_types.index(propeller_type)

                # 检查表格数据是否存在
                if (self.tbl_speed.item(row, 1) is None or
                        self.tbl_speed.item(row, 2) is None or
                        self.tbl_speed.item(row, 4) is None or
                        self.tbl_speed.item(row, 5) is None):
                    QMessageBox.warning(self, "警告", f"型号 {propeller_type} 的最大航速计算结果不完整")
                    continue

                try:
                    vmax = float(self.tbl_speed.item(row, 1).text())
                    p_d = float(self.tbl_speed.item(row, 2).text())
                    D = float(self.tbl_speed.item(row, 4).text())
                    eta0 = float(self.tbl_speed.item(row, 5).text())
                except (ValueError, AttributeError) as e:
                    QMessageBox.warning(self, "数据错误", f"读取型号 {propeller_type} 的数据时出错: {str(e)}")
                    continue

                PD = self.res['PD']
                N = self.res['N']
                w = self.res['w']
                VA = 0.5144 * vmax * (1 - w)
                omega = 0.7 * np.pi * N * D / 60
                V_0_7R_sq = VA ** 2 + omega ** 2
                sigma = (p0_total - pv) / (0.5 * rho * V_0_7R_sq)
                if self.rb_wag.isChecked():
                    tau_c = self.get_tau_c(sigma, source='wag')
                else:
                    tau_c = self.get_tau_c(sigma, source='ber')
                T = PD * eta0 * 1000 / VA
                Ap = T / (0.5 * rho * V_0_7R_sq * tau_c)
                AE = Ap / (1.067 - 0.229 * p_d)
                AE_A0 = AE / (np.pi * D ** 2 / 4)
                self.cavitation_results[propeller_type] = {
                    'AE_A0': AE_A0, 'p_d': p_d, 'D': D, 'eta0': eta0, 'vmax': vmax}
                # 填表
                self.cavitation_table.setItem(0, col, QTableWidgetItem(f"{PD:.1f}"))
                self.cavitation_table.setItem(1, col, QTableWidgetItem(f"{vmax:.2f}"))
                self.cavitation_table.setItem(2, col, QTableWidgetItem(f"{VA:.3f}"))
                self.cavitation_table.setItem(3, col, QTableWidgetItem(f"{omega:.3f}"))
                self.cavitation_table.setItem(4, col, QTableWidgetItem(f"{V_0_7R_sq:.2f}"))
                self.cavitation_table.setItem(5, col, QTableWidgetItem(f"{sigma:.4f}"))
                self.cavitation_table.setItem(6, col, QTableWidgetItem(f"{tau_c:.4f}"))
                self.cavitation_table.setItem(7, col, QTableWidgetItem(f"{T:.0f}"))
                self.cavitation_table.setItem(8, col, QTableWidgetItem(f"{AE_A0:.4f}"))

            if self.cavitation_results:
                self.opt_res = self.cavitation_results[propeller_types[0]]
                self.plot_btn.setEnabled(True)
                self.results_btn.setEnabled(True)
                QMessageBox.information(self, "成功", "空泡校核计算完成")
            else:
                QMessageBox.warning(self, "警告", "空泡校核计算失败，请检查数据")

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"空泡校核计算失败: {str(e)}")

    def plot_curves_and_find_optimum(self):
        if not self.cavitation_results:
            QMessageBox.warning(self, "警告", "请先完成空泡校核计算")
            return

        try:
            if self.blade_count == 4:
                blade_ratios = np.array([0.40, 0.55, 0.70])
            else:
                blade_ratios = np.array([0.50, 0.65, 0.80])

            AE_A0 = np.array([self.cavitation_results[t]['AE_A0'] for t in self.cavitation_results.keys()])
            p_d = np.array([self.cavitation_results[t]['p_d'] for t in self.cavitation_results.keys()])
            D = np.array([self.cavitation_results[t]['D'] for t in self.cavitation_results.keys()])
            eta0 = np.array([self.cavitation_results[t]['eta0'] for t in self.cavitation_results.keys()])
            vmax = np.array([self.cavitation_results[t]['vmax'] for t in self.cavitation_results.keys()])

            x_min, x_max = blade_ratios.min(), blade_ratios.max()
            x_fine = np.linspace(x_min, x_max, 100)

            # 使用曲线拟合而不是简单的线性插值
            try:
                # 使用三次样条插值获得平滑曲线
                f_ae = CubicSpline(blade_ratios, AE_A0)
                f_pd = CubicSpline(blade_ratios, p_d)
                f_d = CubicSpline(blade_ratios, D)
                f_eta = CubicSpline(blade_ratios, eta0)
                f_v = CubicSpline(blade_ratios, vmax)
            except:
                # 如果三次样条失败，使用Akima插值
                f_ae = Akima1DInterpolator(blade_ratios, AE_A0)
                f_pd = Akima1DInterpolator(blade_ratios, p_d)
                f_d = Akima1DInterpolator(blade_ratios, D)
                f_eta = Akima1DInterpolator(blade_ratios, eta0)
                f_v = Akima1DInterpolator(blade_ratios, vmax)

            # 找到交点
            diff = f_ae(x_fine) - x_fine
            idx = np.argmin(np.abs(diff))
            opt_r = x_fine[idx]
            opt_ae = f_ae(opt_r)
            opt_pd = f_pd(opt_r)
            opt_d = f_d(opt_r)
            opt_eta = f_eta(opt_r)
            opt_v = f_v(opt_r)

            self.optimum_results = {'blade_ratio': opt_r, 'AE_A0': opt_ae, 'p_d': opt_pd,
                                    'D': opt_d, 'eta0': opt_eta, 'vmax': opt_v}

            # 创建绘图窗口
            self.plot_window = QDialog(self)
            self.plot_window.setWindowTitle("最佳螺旋桨要素确定")
            self.plot_window.setGeometry(150, 150, 800, 1000)
            fig = Figure(figsize=(8, 10), dpi=100)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self.plot_window)

            # 设置字体大小
            plt.rcParams.update({'font.size': 10})

            # 使用中文标签
            ylabels = ['敞水效率 η₀', '直径 D (m)', '盘面比 AE/A₀',
                       '螺距比 P/D', '最大航速 Vmax (kn)']
            ydatas = [eta0, D, AE_A0, p_d, vmax]
            f_funcs = [f_eta, f_d, f_ae, f_pd, f_v]

            # 定义不同的线型和颜色
            line_styles = ['-', '--', '-.', ':']
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            markers = ['o', 's', '^', 'D', 'v']

            axes = []

            for i in range(5):
                ax = fig.add_subplot(5, 1, i + 1)
                axes.append(ax)

                y_cur = f_funcs[i](x_fine)

                # 绘制平滑曲线 - 使用不同线型和颜色
                ax.plot(x_fine, y_cur, color=colors[i], linestyle=line_styles[i % len(line_styles)],
                        lw=2, alpha=0.8, label='拟合曲线')
                ax.plot(blade_ratios, ydatas[i], color=colors[i], marker=markers[i % len(markers)],
                        markersize=6, linestyle='none', label='数据点')
                ax.axvline(opt_r, color='r', ls='--', lw=2, label=f'最佳值: {opt_r:.3f}')
                ax.set_ylabel(ylabels[i], fontsize=12)
                ax.set_xlim(x_min - 0.05, x_max + 0.05)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best', fontsize=9)

                if i == 4:
                    ax.set_xlabel('盘面比', fontsize=12)

            fig.suptitle('最佳螺旋桨要素确定曲线', fontsize=14, fontweight='bold')
            fig.subplots_adjust(hspace=0.3)
            layout = QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            self.plot_window.setLayout(layout)
            self.plot_window.show()

            self.update_results_text()

        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘制曲线时发生错误: {str(e)}")

    def update_results_text(self):
        if hasattr(self, 'optimum_results'):
            r = self.optimum_results
            text = (f"最佳螺旋桨要素计算结果:\n\n"
                    f"盘面比: {r['blade_ratio']:.4f}\n"
                    f"螺距比 P/D: {r['p_d']:.4f}\n"
                    f"直径 D: {r['D']:.4f} m\n"
                    f"敞水效率 η₀: {r['eta0']:.4f}\n"
                    f"最大航速 Vmax: {r['vmax']:.4f} kn\n\n"
                    "此结果满足桨叶在该工况下不发生空泡的要求。")
            self.result_text.setText(text)

    def show_optimum_results(self):
        if hasattr(self, 'optimum_results'):
            self.update_results_text()
        else:
            QMessageBox.warning(self, "警告", "请先完成最佳要素确定计算")

    # ===================== 3. 强度校核 =====================
    def create_strength_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 8, 8, 8)

        # 强度参数组
        input_group = StyledGroupBox("强度参数")
        form_layout = QFormLayout()
        form_layout.setSpacing(6)

        self.epsilon_label, self.epsilon_input = self.create_styled_input("后倾角系数 ε", "8")
        self.k_coef_label, self.k_coef_input = self.create_styled_input("材料系数 K", "1.0")

        form_layout.addRow(self.epsilon_label, self.epsilon_input)
        form_layout.addRow(self.k_coef_label, self.k_coef_input)

        input_group.setLayout(form_layout)
        lay.addWidget(input_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_str_calc = StyledButton("计算强度")
        self.btn_str_export = StyledButton("导出CSV")

        btn_layout.addWidget(self.btn_str_calc)
        btn_layout.addWidget(self.btn_str_export)
        lay.addLayout(btn_layout)

        # 结果表格
        table_group = StyledGroupBox("强度校核结果")
        table_layout = QVBoxLayout()
        self.tbl_strength = StyledTableWidget(19, 4)
        self.tbl_strength.setHorizontalHeaderLabels(["项目", "0.25R", "0.6R", "单位"])
        table_layout.addWidget(self.tbl_strength)
        table_group.setLayout(table_layout)
        lay.addWidget(table_group)

        self.btn_str_calc.clicked.connect(self.calculate_strength)
        self.btn_str_export.clicked.connect(self.export_strength)

        return w

    def safe_float_convert(self, value, default=0.0):
        """安全转换浮点数，避免空字符串转换错误"""
        try:
            if value is None or str(value).strip() == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def calculate_strength(self):
        # 检查必要的前置计算是否完成
        if not self.opt_res:
            QMessageBox.warning(self, "警告", "请先完成空泡校核计算")
            return

        try:
            # 安全获取输入值
            epsilon = self.safe_float_convert(self.epsilon_input.text(), 8.0)
            K = self.safe_float_convert(self.k_coef_input.text(), 1.0)

            # 安全获取其他必要参数
            D = self.safe_float_convert(self.opt_res.get('D', 0))
            P_D = self.safe_float_convert(self.opt_res.get('p_d', 0))

            if D <= 0 or P_D <= 0:
                QMessageBox.warning(self, "警告", "螺旋桨直径或螺距比数据无效，请先完成空泡校核计算")
                return

            Ad = self.safe_float_convert(self.opt_res.get('AE_A0', 0))

            # 安全获取其他参数
            n_text = self.n_input.text() if hasattr(self, 'n_input') else "155"
            ps_text = self.ps_input.text() if hasattr(self, 'ps_input') else "6222"
            etas_text = self.etas_input.text() if hasattr(self, 'etas_input') else "0.97"

            ne = self.safe_float_convert(n_text, 155)
            Ps = self.safe_float_convert(ps_text, 6222)
            Ne = self.safe_float_convert(etas_text, 0.97) * Ps
            Z = self.blade_count
            G = 7.6

            # 计算弦长
            b_66 = (0.226 * D * Ad) / (0.1 * Z)
            b_025 = 0.7212 * b_66
            b_06 = 0.9911 * b_66

            D_P = 1.0 / P_D if P_D > 0 else 0

            results = {}
            radius_points = [(0.25, b_025), (0.6, b_06)]

            for r_R, b in radius_points:
                if r_R == 0.25:
                    K1, K2, K3, K4 = 634, 250, 1410, 4
                    K5, K6, K7, K8 = 82, 34, 41, 380
                else:
                    K1, K2, K3, K4 = 207, 151, 635, 34
                    K5, K6, K7, K8 = 23, 12, 65, 330

                # 计算A1和Y
                A1 = D_P * (K1 - K2 * D_P) + K3 * D_P - K4
                Y = (1.36 * A1 * Ne) / (Z * b * ne) if (Z * b * ne) > 0 else 0

                # 计算A2和X
                A2 = D_P * (K5 + K6 * epsilon) + K7 * epsilon + K8
                X = (A2 * G * Ad * ne ** 2 * D ** 3) / (1e10 * Z * b) if (Z * b) > 0 else 0

                # 计算厚度
                t_req = np.sqrt(Y / (K - X)) if (K - X) > 0 and Y > 0 else 0

                # 标准厚度
                if r_R == 0.25:
                    t_std = ((4.06 + 3.59) / 2) * D * 10
                else:
                    t_std = 2.18 * D * 10

                t_actual = max(t_std, t_req) if t_std < t_req else t_std

                results[r_R] = {
                    'b': b, 'A1': A1, 'Y': Y, 'A2': A2, 'X': X,
                    't_req': t_req, 't_std': t_std, 't_actual': t_actual,
                    'conclusion': "满足" if t_std >= t_req else "不满足"
                }

            # 填充表格
            rows = [
                ("弦长 b", results[0.25]['b'], results[0.6]['b'], "m"),
                ("系数 K1", 634, 207, ""),
                ("系数 K2", 250, 151, ""),
                ("系数 K3", 1410, 635, ""),
                ("系数 K4", 4, 34, ""),
                ("A1", results[0.25]['A1'], results[0.6]['A1'], ""),
                ("Y", results[0.25]['Y'], results[0.6]['Y'], "N"),
                ("系数 K5", 82, 23, ""),
                ("系数 K6", 34, 12, ""),
                ("系数 K7", 41, 65, ""),
                ("系数 K8", 380, 330, ""),
                ("A2", results[0.25]['A2'], results[0.6]['A2'], ""),
                ("材料系数 K", K, K, ""),
                ("X", results[0.25]['X'], results[0.6]['X'], "N"),
                ("规范最小厚度", results[0.25]['t_req'], results[0.6]['t_req'], "mm"),
                ("MAU标准厚度", results[0.25]['t_std'], results[0.6]['t_std'], "mm"),
                ("校核结果", results[0.25]['conclusion'], results[0.6]['conclusion'], ""),
                ("实取厚度", results[0.25]['t_actual'], results[0.6]['t_actual'], "mm")
            ]

            self.tbl_strength.setRowCount(len(rows))
            for r, (n, v25, v60, u) in enumerate(rows):
                self.tbl_strength.setItem(r, 0, QTableWidgetItem(n))
                self.tbl_strength.setItem(r, 1, QTableWidgetItem(f"{v25:.4f}" if isinstance(v25, float) else str(v25)))
                self.tbl_strength.setItem(r, 2, QTableWidgetItem(f"{v60:.4f}" if isinstance(v60, float) else str(v60)))
                self.tbl_strength.setItem(r, 3, QTableWidgetItem(u))

            QMessageBox.information(self, "成功", "强度校核计算完成")

        except Exception as e:
            QMessageBox.critical(self, "强度计算错误", f"计算异常: {str(e)}")

    def export_strength(self):
        try:
            path, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "CSV文件 (*.csv)")
            if not path:
                return
            with open(path, 'w', newline='', encoding='gbk') as f:
                writer = csv.writer(f)
                writer.writerow(["项目", "0.25R", "0.6R", "单位"])
                for r in range(self.tbl_strength.rowCount()):
                    row_data = []
                    for c in range(4):
                        item = self.tbl_strength.item(r, c)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            QMessageBox.information(self, "成功", f"已导出到 {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"错误: {str(e)}")

    def create_pitch_correction_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 8, 8, 8)

        # 螺距修正参数
        input_group = StyledGroupBox("螺距修正参数")
        form_layout = QFormLayout()
        form_layout.setSpacing(6)

        self.pc_dhD_label, self.pc_dhD_input = self.create_styled_input("实际毂径比 dh/D", "0.18")
        form_layout.addRow(self.pc_dhD_label, self.pc_dhD_input)

        input_group.setLayout(form_layout)
        lay.addWidget(input_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_pc = StyledButton("螺距修正")
        btn_layout.addWidget(self.btn_pc)
        lay.addLayout(btn_layout)

        # 结果显示
        result_group = StyledGroupBox("螺距修正结果")
        result_layout = QVBoxLayout()
        self.txt_pc_result = StyledTextEdit()
        self.txt_pc_result.setReadOnly(True)
        result_layout.addWidget(self.txt_pc_result)
        result_group.setLayout(result_layout)
        lay.addWidget(result_group)

        self.btn_pc.clicked.connect(self.calculate_pitch_correction)
        return w

    def calculate_pitch_correction(self):
        if not (self.res and (self.opt_res or hasattr(self, 'optimum_results'))):
            QMessageBox.warning(self, "警告", "请先完成最大航速和空泡校核或最佳要素确定计算")
            return

        # 检查是否有最佳要素确定的结果
        if not hasattr(self, 'optimum_results') or not self.optimum_results:
            QMessageBox.warning(self, "警告", "请先完成最佳要素确定计算")
            return

        try:
            # 获取基本参数 - 使用最佳要素确定后的结果
            Vmax = self.optimum_results['vmax']
            Ad = self.optimum_results['AE_A0']  # 最佳盘面比
            PoD = self.optimum_results['p_d']  # 最佳螺距比
            D = self.optimum_results['D']  # 最佳直径

            # 其他参数从原始结果获取
            N = self.res['N']
            Z = self.blade_count

            # 获取毂径比
            dhD_text = self.pc_dhD_input.text().strip() or "0.18"
            dhD = float(dhD_text)

            print(f"螺距修正参数 - 使用最佳要素结果:")
            print(f"PoD={PoD:.4f}, D={D:.3f}, Ad={Ad:.4f}, Vmax={Vmax:.2f}, N={N}")

            # 计算设计桨在0.7R处的厚度和弦长
            # 使用MAU型值表中的厚度百分比数据
            t_02_pct = MAU_THICKNESS['0.2R']  # 4.06%
            t_06_pct = MAU_THICKNESS['0.6R']  # 2.18%
            t_07_pct = MAU_THICKNESS['0.7R']  # 1.71%

            # 转换为实际厚度(mm)
            t_02 = (t_02_pct / 100.0) * D * 1000  # mm
            t_06 = (t_06_pct / 100.0) * D * 1000  # mm
            t_07 = (t_07_pct / 100.0) * D * 1000  # mm

            # 计算0.7R处的弦长
            b_ref_066 = 0.226 * D * Ad / (0.1 * Z)  # 0.66R参考弦长(m)
            b_07_pct = MAU_WIDTH['0.7R']  # 99.64%
            b_07 = (b_07_pct / 100.0) * b_ref_066  # 0.7R弦长(m)

            # 设计桨的[t/b]0.7 (单位: m/m)
            tob_des = (t_07 / 1000.0) / b_07  # 转换为米后计算

            # 标准桨的[t/b]0.7 (使用标准盘面比0.55)
            b_ref_066_std = 0.226 * D * 0.55 / (0.1 * Z)
            b_07_std = (b_07_pct / 100.0) * b_ref_066_std
            tob_std = (t_07 / 1000.0) / b_07_std

            # 厚度修正量
            delta_tob = (tob_des - tob_std) * 0.75

            # 计算滑脱比相关参数
            VA = 0.5144 * Vmax * (1 - self.res['w'])  # 进速(m/s)
            P = PoD * D  # 螺距(m)

            # 计算滑脱比 1-s = VA / (P * n)
            # n = N / 60 (rps)
            n = N / 60.0  # rps
            if P * n > 0:
                one_minus_s = VA / (P * n)
            else:
                one_minus_s = 0

            print(f"滑脱比计算: VA={VA:.3f} m/s, P={P:.3f} m, n={n:.3f} rps, 1-s={one_minus_s:.3f}")

            # 厚度修正引起的螺距比变化
            # 修正公式: Δ(P/D)_t = -2 * (P/D) * (1-s) * Δ(t/b)
            delta_PoD_t = -2 * PoD * one_minus_s * delta_tob

            # 毂径比修正
            # Δ(P/D)_h = (1/10) * (dh/D - 0.18)
            delta_PoD_h = 0.0 if abs(dhD - 0.18) < 1e-6 else (1.0 / 10.0) * (dhD - 0.18)

            # 总修正量
            delta_PoD_total = delta_PoD_t + delta_PoD_h

            # 修正后的螺距比
            PoD_corrected = PoD + delta_PoD_total

            print(
                f"修正量: Δtob={delta_tob:.6f}, ΔPoD_t={delta_PoD_t:.6f}, ΔPoD_h={delta_PoD_h:.6f}, ΔPoD_total={delta_PoD_total:.6f}")
            print(f"螺距比: 原值={PoD:.4f}, 修正后={PoD_corrected:.4f}")

            # 生成报告
            report = (f"螺距修正计算结果：\n\n"
                      f"设计参数（使用最佳要素确定结果）：\n"
                      f"- 螺旋桨直径 D = {D:.3f} m\n"
                      f"- 最佳螺距比 P/D = {PoD:.4f}\n"
                      f"- 最佳盘面比 Ae/Ao = {Ad:.4f}\n"
                      f"- 桨叶数 Z = {Z}\n"
                      f"- 最大航速 Vmax = {Vmax:.2f} kn\n"
                      f"- 主机转速 N = {N} rpm\n"
                      f"- 毂径比 dh/D = {dhD:.3f}\n\n"

                      f"厚度修正计算：\n"
                      f"1) 设计桨0.2R厚度 t₀.₂ = {t_02:.1f} mm\n"
                      f"2) 设计桨0.6R厚度 t₀.₆ = {t_06:.1f} mm\n"
                      f"3) 设计桨0.7R厚度 t₀.₇ = {t_07:.1f} mm\n"
                      f"4) 设计桨0.7R弦长 b₀.₇ = {b_07:.4f} m\n"
                      f"5) 设计桨[t/b]₀.₇ = {tob_des:.6f}\n"
                      f"6) 标准桨[t/b]₀.₇ = {tob_std:.6f}\n"
                      f"7) Δ[t/b]₀.₇ = {delta_tob:.6f}\n\n"

                      f"滑脱比计算：\n"
                      f"8) 进速 VA = {VA:.3f} m/s\n"
                      f"9) 螺距 P = {P:.3f} m\n"
                      f"10) 转速 n = {n:.3f} rps\n"
                      f"11) 滑脱比 1-s = {one_minus_s:.4f}\n\n"

                      f"修正量计算：\n"
                      f"12) 厚度修正 Δ(P/D)ₜ = {delta_PoD_t:.6f}\n"
                      f"13) 毂径比修正 Δ(P/D)ₕ = {delta_PoD_h:.6f}\n"
                      f"14) 总修正量 Δ(P/D) = {delta_PoD_total:.6f}\n"
                      f"15) 修正后螺距比 (P/D)' = {PoD_corrected:.4f}\n\n"

                      f"验证：原最佳P/D {PoD:.4f} + 修正量 {delta_PoD_total:.6f} = {PoD_corrected:.4f}")

            self.txt_pc_result.setText(report)

        except Exception as e:
            QMessageBox.critical(self, "螺距修正错误", f"计算错误: {str(e)}")
            import traceback
            traceback.print_exc()

    # ===================== 5. 质量及惯性矩 =====================

    def create_mass_inertia_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(8, 8, 8, 8)

        # 质量参数组
        input_group = StyledGroupBox("质量及惯性矩计算参数")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(6)

        # 添加桨毂长度和轴径输入
        self.mass_dhD = StyledLineEdit("0.18")
        self.mass_hub_length = StyledLineEdit("0.2")  # 桨毂长度 Lk (m)
        self.mass_shaft_diameter = StyledLineEdit("0.15")  # 轴径 d0 (m)
        self.mass_rho = StyledLineEdit("8400")  # 材料密度，根据文档改为8400 kg/m³
        self.mass_K = StyledLineEdit("1.0")  # 材料系数 K
        self.mass_Z = StyledLineEdit(str(self.blade_count))

        grid_layout.addWidget(QLabel("毂径比 d/D"), 0, 0)
        grid_layout.addWidget(self.mass_dhD, 0, 1)
        grid_layout.addWidget(QLabel("桨毂长度 Lk (m)"), 0, 2)
        grid_layout.addWidget(self.mass_hub_length, 0, 3)
        grid_layout.addWidget(QLabel("轴径 d0 (m)"), 1, 0)
        grid_layout.addWidget(self.mass_shaft_diameter, 1, 1)
        grid_layout.addWidget(QLabel("材料密度 ρ (kg/m³)"), 1, 2)
        grid_layout.addWidget(self.mass_rho, 1, 3)
        grid_layout.addWidget(QLabel("材料系数 K"), 2, 0)
        grid_layout.addWidget(self.mass_K, 2, 1)
        grid_layout.addWidget(QLabel("桨叶数量 Z"), 2, 2)
        grid_layout.addWidget(self.mass_Z, 2, 3)

        input_group.setLayout(grid_layout)
        lay.addWidget(input_group)

        # 按钮
        btn_calc = StyledButton("计算质量及惯性矩")
        btn_calc.clicked.connect(self.calculate_mass_properties)
        lay.addWidget(btn_calc)

        # 结果标签页
        tabs = QTabWidget()

        # 汇总结果标签页
        sum_w = QWidget()
        v = QVBoxLayout(sum_w)
        self.tbl_mass_results = StyledTableWidget(20, 4)
        self.tbl_mass_results.setHorizontalHeaderLabels(["参数", "数值", "单位", "公式"])
        v.addWidget(self.tbl_mass_results)
        tabs.addTab(sum_w, "汇总结果")

        # 详细计算标签页
        det_w = QWidget()
        v2 = QVBoxLayout(det_w)
        self.tbl_mass_details = StyledTableWidget(0, 9)
        self.tbl_mass_details.setHorizontalHeaderLabels([
            "半径位置", "r/R", "面积系数Ka", "b×t", "切面面积S",
            "辛氏系数SM", "4×5", "R", "R²", "6×7", "6×8"
        ])
        v2.addWidget(self.tbl_mass_details)
        tabs.addTab(det_w, "详细计算")

        lay.addWidget(tabs)

        # 导出按钮
        btn_exp = StyledButton("导出结果")
        btn_exp.clicked.connect(self.export_mass_details)
        lay.addWidget(btn_exp)

        return w

    def calculate_mass_properties(self):
        """根据图片中的公式重新实现质量及惯性矩计算"""
        try:
            if not (self.opt_res or hasattr(self, 'optimum_results')):
                QMessageBox.warning(self, "警告", "请先完成空泡校核或最佳要素确定计算")
                return

            # 安全获取输入值
            d_D = self.safe_float_convert(self.mass_dhD.text(), 0.18)
            hub_length = self.safe_float_convert(self.mass_hub_length.text(), 0.2)  # 桨毂长度 Lk
            shaft_diameter = self.safe_float_convert(self.mass_shaft_diameter.text(), 0.15)  # 轴径
            rho = self.safe_float_convert(self.mass_rho.text(), 8400)  # 材料密度
            Z = int(self.safe_float_convert(self.mass_Z.text(), self.blade_count))
            K = self.safe_float_convert(self.mass_K.text(), 1.0)  # 材料系数 K

            # 获取螺旋桨基本参数 - 优先使用最佳要素确定的结果
            if hasattr(self, 'optimum_results') and self.optimum_results:
                # 使用最佳要素确定的结果
                D = self.safe_float_convert(self.optimum_results.get('D', 0))
                Ae_Ao = self.safe_float_convert(self.optimum_results.get('AE_A0', 0))
                print(f"使用最佳要素确定结果计算质量惯性矩: D={D}m, Ae/Ao={Ae_Ao}")
            else:
                # 使用空泡校核结果
                D = self.safe_float_convert(self.opt_res.get('D', 0))
                Ae_Ao = self.safe_float_convert(self.opt_res.get('AE_A0', 0))
                print(f"使用空泡校核结果计算质量惯性矩: D={D}m, Ae/Ao={Ae_Ao}")

            if D <= 0 or Ae_Ao <= 0:
                QMessageBox.warning(self, "警告", "螺旋桨直径或盘面比数据无效")
                return

            # 获取功率和转速参数
            if hasattr(self, 'res'):
                PD = self.res.get('PD', 0)  # 推进功率 kW
                N = self.res.get('N', 0)  # 转速 rpm
            else:
                PD = 0
                N = 0

            print(f"计算参数: D={D}m, Ae/Ao={Ae_Ao}, Z={Z}, ρ={rho}kg/m³, PD={PD}kW, N={N}rpm, K={K}")

            # 计算参考弦长（0.66R处的弦长）- 即最大宽度
            b_max = 0.226 * D * Ae_Ao / (0.1 * Z)
            print(f"桨叶最大宽度 b_max: {b_max:.4f}m")

            # 计算桨毂直径
            hub_diameter = d_D * D  # 桨毂直径 d
            print(f"桨毂直径: {hub_diameter:.4f}m")

            # 计算桨轴中央处轴径 d0
            # 公式: d0 = 0.045 + 0.12(P_D/N)^(1/3) - (K * Lk) / 2
            if PD > 0 and N > 0:
                d0 = 0.045 + 0.12 * (PD / N) ** (1 / 3) - (K * hub_length) / 2
            else:
                # 使用默认计算
                d0 = (1 / 13) * hub_length * 2

            # 确保d0不为负值
            d0 = max(0.01, d0)

            print(f"桨轴中央处轴径 d0: {d0:.4f}m (K={K}, Lk={hub_length}m)")

            # 获取0.2R和0.6R处的厚度
            t_02_pct = MAU_THICKNESS['0.2R']  # 4.06%
            t_06_pct = MAU_THICKNESS['0.6R']  # 2.18%
            t_02 = (t_02_pct / 100.0) * D  # 转换为实际厚度(m)
            t_06 = (t_06_pct / 100.0) * D  # 转换为实际厚度(m)

            print(f"0.2R厚度: {t_02:.4f}m, 0.6R厚度: {t_06:.4f}m")

            # 根据图片中的公式计算桨叶质量
            # M_b1 = 0.169 * ρ * Z * b_max * (0.5*t_0.2 + t_0.6) * (1 - d/D) * D
            blade_mass = 0.169 * rho * Z * b_max * (0.5 * t_02 + t_06) * (1 - d_D) * D

            # 计算桨毂质量
            # M_n = [0.88 - 0.6*(d0/d)] * Lk * ρ * d²
            d0_d_ratio = d0 / hub_diameter if hub_diameter > 0 else 0
            coeff = 0.88 - 0.6 * d0_d_ratio
            coeff = max(0.1, min(1.0, coeff))  # 限制在合理范围内
            hub_mass = coeff * hub_length * rho * (hub_diameter ** 2)

            # 总质量
            total_mass = blade_mass + hub_mass

            print(f"桨叶质量: {blade_mass:.2f}kg, 桨毂质量: {hub_mass:.2f}kg, 总质量: {total_mass:.2f}kg")

            # 计算螺旋桨质量惯性矩 - 根据d/D选择不同公式
            if d_D <= 0.18:
                # 当 d/D ≤ 0.18 时
                # I_mp = 0.0948 * ρ * Z * b_max * (0.5*t_0.2 + t_0.6) * D³
                inertia = 0.0948 * rho * Z * b_max * (0.5 * t_02 + t_06) * (D ** 3)
                inertia_formula = "I_mp = 0.0948·ρ·Z·b_max·(0.5t₀₂+t₀₆)·D³ (d/D ≤ 0.18)"
            else:
                # 当 d/D > 0.18 时
                # I_mp = [0.0648 + 0.167·d/D]·ρ·Z·b_max·(0.5t₀₂+t₀₆)·D³
                inertia = (0.0648 + 0.167 * d_D) * rho * Z * b_max * (0.5 * t_02 + t_06) * (D ** 3)
                inertia_formula = f"I_mp = [0.0648+0.167·d/D]·ρ·Z·b_max·(0.5t₀₂+t₀₆)·D³ (d/D > 0.18)"

            print(f"螺旋桨质量惯性矩: {inertia:.2f} kg·m²")

            # 更新结果表格
            results = [
                ("螺旋桨直径 D", f"{D:.4f}", "m", "D = 2R"),
                ("桨叶数量 Z", f"{Z}", "", ""),
                ("材料密度 ρ", f"{rho:.0f}", "kg/m³", "输入值"),
                ("材料系数 K", f"{K:.3f}", "", "输入值"),
                ("盘面比 Ae/Ao", f"{Ae_Ao:.4f}", "", ""),
                ("毂径比 d/D", f"{d_D:.4f}", "", "输入值"),
                ("桨叶最大宽度 b_max", f"{b_max:.4f}", "m", "0.66R处弦长"),
                ("0.2R厚度 t₀₂", f"{t_02:.4f}", "m", f"{t_02_pct}% × D"),
                ("0.6R厚度 t₀₆", f"{t_06:.4f}", "m", f"{t_06_pct}% × D"),
                ("桨毂直径 d", f"{hub_diameter:.4f}", "m", "d = d/D × D"),
                ("桨毂长度 Lk", f"{hub_length:.3f}", "m", "输入值"),
                ("桨轴中央处轴径 d0", f"{d0:.4f}", "m", "d0 = 0.045 + 0.12(PD/N)^(1/3) - (K×Lk)/2"),
                ("d0/d", f"{d0_d_ratio:.4f}", "", "d0/d"),
                ("桨叶质量 M_b1", f"{blade_mass:.2f}", "kg", "0.169·ρ·Z·b_max·(0.5t₀₂+t₀₆)·(1-d/D)·D"),
                ("桨毂质量 M_n", f"{hub_mass:.2f}", "kg", "[0.88-0.6·(d0/d)]·Lk·ρ·d²"),
                ("螺旋桨总质量 M", f"{total_mass:.2f}", "kg", "M_b1 + M_n"),
                ("质量惯性矩 I_mp", f"{inertia:.2f}", "kg·m²", inertia_formula)
            ]

            self.tbl_mass_results.setRowCount(len(results))
            for r, (param, value, unit, formula) in enumerate(results):
                self.tbl_mass_results.setItem(r, 0, QTableWidgetItem(param))
                self.tbl_mass_results.setItem(r, 1, QTableWidgetItem(value))
                self.tbl_mass_results.setItem(r, 2, QTableWidgetItem(unit))
                self.tbl_mass_results.setItem(r, 3, QTableWidgetItem(formula))

            # 更新详细计算表格（使用辛普森法的详细计算）
            self.update_mass_details_table(D, Ae_Ao, Z, rho)

            QMessageBox.information(self, "成功", "质量及惯性矩计算完成")

        except Exception as e:
            print(f"质量计算错误: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "计算错误", f"质量计算失败: {str(e)}")

    def update_mass_details_table(self, D, Ae_Ao, Z, rho):
        """更新详细计算表格（辛普森法）"""
        # 计算参考弦长
        b_ref_066 = 0.226 * D * Ae_Ao / (0.1 * Z)

        # 计算每个半径位置的参数
        r_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        self.mass_details = []
        total_4x5 = 0
        total_6x7 = 0
        total_6x8 = 0

        for r_R in r_positions:
            position_key = f'{r_R:.1f}R'

            # 获取型值数据
            t_pct = MAU_THICKNESS[position_key]  # 厚度百分比
            b_pct = MAU_WIDTH[position_key]  # 宽度百分比
            Ka = AREA_COEFF[position_key]  # 面积系数
            SM = SIMPSON_COEFF[position_key]  # 辛普森系数

            # 计算实际尺寸（单位：米）
            t_actual = (t_pct / 100.0) * D  # 厚度，单位：m
            b_actual = (b_pct / 100.0) * b_ref_066  # 弦长，单位：m

            # 计算各项参数
            b_t = b_actual * t_actual  # b×t
            section_area = b_t * Ka  # 切面面积S
            col_4x5 = section_area * SM  # 4×5
            col_6x7 = col_4x5 * r_R  # 6×7
            col_6x8 = col_4x5 * (r_R ** 2)  # 6×8

            # 累加
            total_4x5 += col_4x5
            total_6x7 += col_6x7
            total_6x8 += col_6x8

            self.mass_details.append({
                'position': position_key,
                'r_R': r_R,
                'Ka': Ka,
                'b_t': b_t,
                'section_area': section_area,
                'SM': SM,
                'col_4x5': col_4x5,
                'col_6x7': col_6x7,
                'col_6x8': col_6x8
            })

        # 更新详细计算表格
        self.tbl_mass_details.setRowCount(len(self.mass_details) + 1)

        for row, detail in enumerate(self.mass_details):
            self.tbl_mass_details.setItem(row, 0, QTableWidgetItem(detail['position']))
            self.tbl_mass_details.setItem(row, 1, QTableWidgetItem(f"{detail['r_R']:.1f}"))
            self.tbl_mass_details.setItem(row, 2, QTableWidgetItem(f"{detail['Ka']:.4f}"))
            self.tbl_mass_details.setItem(row, 3, QTableWidgetItem(f"{detail['b_t']:.4f}"))
            self.tbl_mass_details.setItem(row, 4, QTableWidgetItem(f"{detail['section_area']:.4f}"))
            self.tbl_mass_details.setItem(row, 5, QTableWidgetItem(str(detail['SM'])))
            self.tbl_mass_details.setItem(row, 6, QTableWidgetItem(f"{detail['col_4x5']:.4f}"))
            self.tbl_mass_details.setItem(row, 7, QTableWidgetItem(f"{detail['r_R']:.1f}"))
            self.tbl_mass_details.setItem(row, 8, QTableWidgetItem(f"{detail['r_R'] ** 2:.2f}"))
            self.tbl_mass_details.setItem(row, 9, QTableWidgetItem(f"{detail['col_6x7']:.4f}"))
            self.tbl_mass_details.setItem(row, 10, QTableWidgetItem(f"{detail['col_6x8']:.4f}"))

        # 添加汇总行
        summary_row = len(self.mass_details)
        self.tbl_mass_details.setItem(summary_row, 0, QTableWidgetItem("辛普森求和"))
        self.tbl_mass_details.setItem(summary_row, 6, QTableWidgetItem(f"{total_4x5:.4f}"))
        self.tbl_mass_details.setItem(summary_row, 9, QTableWidgetItem(f"{total_6x7:.4f}"))
        self.tbl_mass_details.setItem(summary_row, 10, QTableWidgetItem(f"{total_6x8:.4f}"))

    def export_mass_details(self):
        try:
            if not self.mass_details:
                QMessageBox.warning(self, "警告", "无数据可导出")
                return
            path, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "CSV文件 (*.csv)")
            if not path:
                return
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["半径位置", "r/R", "面积系数Ka", "b×t", "切面面积S",
                                 "辛氏系数SM", "4×5", "R", "R²", "6×7", "6×8"])
                for d in self.mass_details:
                    writer.writerow([
                        d['position'], f"{d['r_R']:.1f}", f"{d['Ka']:.4f}",
                        f"{d['b_t']:.4f}", f"{d['section_area']:.4f}", d['SM'],
                        f"{d['col_4x5']:.4f}", f"{d['r_R']:.1f}", f"{d['r_R'] ** 2:.2f}",
                        f"{d['col_6x7']:.4f}", f"{d['col_6x8']:.4f}"
                    ])
            QMessageBox.information(self, "成功", f"已导出到 {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"错误: {str(e)}")

    # ===================== 6. 敞水曲线 =====================
    def create_open_water_tab(self):
        """创建敞水曲线标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 输入参数组
        input_group = StyledGroupBox("输入参数")
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)
        input_layout.setContentsMargins(8, 10, 8, 10)

        # 桨叶数
        blade_layout = QVBoxLayout()
        blade_label = QLabel("桨叶数")
        blade_label.setStyleSheet("font-weight: 600;")
        blade_layout.addWidget(blade_label)
        self.plot_blade_spin = QSpinBox()
        self.plot_blade_spin.setRange(4, 5)
        self.plot_blade_spin.setValue(self.blade_count)
        self.plot_blade_spin.valueChanged.connect(self.on_plot_blade_count_changed)
        blade_layout.addWidget(self.plot_blade_spin)
        input_layout.addLayout(blade_layout)

        # 盘面比 (AE/AO)
        area_ratio_layout = QVBoxLayout()
        area_ratio_label = QLabel("盘面比 (AE/AO)")
        area_ratio_label.setStyleSheet("font-weight: 600;")
        area_ratio_layout.addWidget(area_ratio_label)
        self.plot_area_ratio_spin = QDoubleSpinBox()
        self.plot_area_ratio_spin.setRange(0.3, 1.2)
        self.plot_area_ratio_spin.setSingleStep(0.0001)
        self.plot_area_ratio_spin.setDecimals(4)
        self.plot_area_ratio_spin.setValue(0.55)
        area_ratio_layout.addWidget(self.plot_area_ratio_spin)
        input_layout.addLayout(area_ratio_layout)

        # 螺距比 (P/D)
        pitch_ratio_layout = QVBoxLayout()
        pitch_ratio_label = QLabel("螺距比 (P/D)")
        pitch_ratio_label.setStyleSheet("font-weight: 600;")
        pitch_ratio_layout.addWidget(pitch_ratio_label)
        self.plot_pitch_ratio_spin = QDoubleSpinBox()
        self.plot_pitch_ratio_spin.setRange(0, 2.0)
        self.plot_pitch_ratio_spin.setSingleStep(0.0001)
        self.plot_pitch_ratio_spin.setDecimals(4)
        self.plot_pitch_ratio_spin.setValue(0.8)
        pitch_ratio_layout.addWidget(self.plot_pitch_ratio_spin)
        input_layout.addLayout(pitch_ratio_layout)

        input_group.setLayout(input_layout)

        # 图表设置组
        plot_settings_group = StyledGroupBox("图表设置")
        plot_settings_layout = QHBoxLayout()
        plot_settings_layout.setSpacing(12)
        plot_settings_layout.setContentsMargins(8, 10, 8, 10)

        # J范围设置
        j_range_layout = QVBoxLayout()
        j_range_label = QLabel("J范围")
        j_range_label.setStyleSheet("font-weight: 600;")
        j_range_layout.addWidget(j_range_label)
        j_range_sub_layout = QHBoxLayout()
        self.j_min_spin = QDoubleSpinBox()
        self.j_min_spin.setRange(0, 1.5)
        self.j_min_spin.setSingleStep(0.1)
        self.j_min_spin.setValue(0.0)
        j_range_sub_layout.addWidget(self.j_min_spin)
        j_range_sub_layout.addWidget(QLabel("到"))
        self.j_max_spin = QDoubleSpinBox()
        self.j_max_spin.setRange(0.1, 2.0)
        self.j_max_spin.setSingleStep(0.1)
        self.j_max_spin.setValue(1.6)
        j_range_sub_layout.addWidget(self.j_max_spin)
        j_range_layout.addLayout(j_range_sub_layout)
        plot_settings_layout.addLayout(j_range_layout)

        # 步长设置
        step_layout = QVBoxLayout()
        step_label = QLabel("步长")
        step_label.setStyleSheet("font-weight: 600;")
        step_layout.addWidget(step_label)
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.01, 0.5)
        self.step_spin.setSingleStep(0.01)
        self.step_spin.setValue(0.1)
        step_layout.addWidget(self.step_spin)
        plot_settings_layout.addLayout(step_layout)

        plot_settings_group.setLayout(plot_settings_layout)

        # 按钮组
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        plot_btn = StyledButton("生成曲线")
        plot_btn.clicked.connect(self.generate_plot)
        btn_layout.addWidget(plot_btn)

        save_btn = StyledButton("保存图片")
        save_btn.clicked.connect(self.save_plot)
        btn_layout.addWidget(save_btn)

        # 图表区域
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)

        # 添加到主布局
        layout.addWidget(input_group)
        layout.addWidget(plot_settings_group)
        layout.addLayout(btn_layout)
        layout.addWidget(self.canvas)

        return tab

    def on_plot_blade_count_changed(self):
        """当敞水曲线页面的桨叶数改变时更新系数"""
        blade_num = self.plot_blade_spin.value()
        self.au_coeffs.update_coefficients_by_blade_count(blade_num)

    def generate_plot(self):
        """生成敞水性能曲线"""
        blade_num = self.plot_blade_spin.value()
        if not self.au_coeffs.update_coefficients_by_blade_count(blade_num):
            QMessageBox.warning(self, "警告", f"暂不支持{blade_num}叶桨的计算")
            return

        area_ratio = self.plot_area_ratio_spin.value()
        pitch_ratio = self.plot_pitch_ratio_spin.value()

        # 获取J范围和步长
        j_min = self.j_min_spin.value()
        j_max = self.j_max_spin.value()
        step = self.step_spin.value()

        # 生成J值序列
        j_values = np.arange(j_min, j_max + step, step)

        # 计算KT, 10KQ和η0
        kt_values = []
        ten_kq_values = []
        eta0_values = []

        for j in j_values:
            # 计算KT
            kt = np.float64(self.au_coeffs.current_kt_coeffs[0]['value'])
            for coeff in self.au_coeffs.current_kt_coeffs[1:]:
                term = np.float64(coeff['value'])
                if coeff['i'] > 0:
                    term *= np.power(np.float64(pitch_ratio), np.int32(coeff['i']))
                if coeff['j'] > 0:
                    term *= np.power(np.float64(j), np.int32(coeff['j']))
                if coeff['k'] > 0:
                    term *= np.power(np.float64(area_ratio), np.int32(coeff['k']))
                kt += term
            kt_values.append(kt)

            # 计算10KQ
            ten_kq = np.float64(self.au_coeffs.current_kq_coeffs[0]['value'])
            for coeff in self.au_coeffs.current_kq_coeffs[1:]:
                term = np.float64(coeff['value'])
                if coeff['i'] > 0:
                    term *= np.power(np.float64(pitch_ratio), np.int32(coeff['i']))
                if coeff['j'] > 0:
                    term *= np.power(np.float64(j), np.int32(coeff['j']))
                if coeff['k'] > 0:
                    term *= np.power(np.float64(area_ratio), np.int32(coeff['k']))
                ten_kq += term
            ten_kq_values.append(ten_kq)

            # 计算KQ (10KQ / 10)
            kq = ten_kq / 10.0

            # 计算敞水效率
            if j != 0 and kq != 0:
                eta0 = (kt * j) / (2 * np.pi * kq)
            else:
                eta0 = 0.0
            eta0_values.append(eta0)

        # 绘制图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 定义不同的线型和标记
        line_styles = ['-', '--', '-.']
        markers = ['o', 's', '^']

        # 绘制KT曲线 - 使用不同线型
        ax.plot(j_values, kt_values, 'b-', linewidth=2, label='KT', linestyle=line_styles[0])

        # 绘制10KQ曲线 - 使用不同线型
        ax.plot(j_values, ten_kq_values, 'r-', linewidth=2, label='10KQ', linestyle=line_styles[1])

        # 绘制η0曲线
        ax2 = ax.twinx()
        ax2.plot(j_values, eta0_values, 'g-', linewidth=2, label='η0', linestyle=line_styles[2])

        # 设置标题和标签 - 使用中文
        ax.set_title(f"AU{blade_num}-{area_ratio:.2f} 螺距比(P/D)={pitch_ratio:.2f} 敞水性能曲线",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('进速系数 J', fontsize=11)
        ax.set_ylabel('KT, 10KQ', fontsize=11)
        ax2.set_ylabel('敞水效率 η0', fontsize=11)

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)

        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

        # 刷新画布
        self.canvas.draw()

    def save_plot(self):
        """保存图表为图片"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "", "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "保存成功", f"图片已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存图片时发生错误:\n{str(e)}")

    # ===================== 7. 系柱计算 =====================
    def create_mooring_tab(self):
        """创建系柱计算标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 输入参数组
        input_group = StyledGroupBox("输入参数")
        input_layout = QGridLayout()
        input_layout.setSpacing(6)
        input_layout.setContentsMargins(8, 10, 8, 10)

        # 主机功率
        input_layout.addWidget(QLabel("主机功率 Ps (kW)"), 0, 0)
        self.mooring_ps = StyledLineEdit()
        self.mooring_ps.setPlaceholderText("从最大航速计算获取")
        input_layout.addWidget(self.mooring_ps, 0, 1)

        # 主机转速
        input_layout.addWidget(QLabel("主机转速 N (r/min)"), 1, 0)
        self.mooring_n = StyledLineEdit()
        self.mooring_n.setPlaceholderText("从最大航速计算获取")
        input_layout.addWidget(self.mooring_n, 1, 1)

        # 轴系效率
        input_layout.addWidget(QLabel("轴系效率 ηS"), 2, 0)
        self.mooring_eta_s = StyledLineEdit()
        self.mooring_eta_s.setPlaceholderText("从最大航速计算获取")
        input_layout.addWidget(self.mooring_eta_s, 2, 1)

        # 相对旋转效率
        input_layout.addWidget(QLabel("相对旋转效率 ηR"), 3, 0)
        self.mooring_eta_r = StyledLineEdit()
        self.mooring_eta_r.setPlaceholderText("从最大航速计算获取")
        input_layout.addWidget(self.mooring_eta_r, 3, 1)

        # 推力减额分数
        input_layout.addWidget(QLabel("推力减额分数 t0"), 4, 0)
        self.mooring_t0 = StyledLineEdit("0.04")
        input_layout.addWidget(self.mooring_t0, 4, 1)

        # 螺旋桨直径
        input_layout.addWidget(QLabel("螺旋桨直径 D (m)"), 5, 0)
        self.mooring_d = StyledLineEdit()
        self.mooring_d.setPlaceholderText("从空泡校核获取")
        input_layout.addWidget(self.mooring_d, 5, 1)

        # J=0时的KT和KQ
        input_layout.addWidget(QLabel("J=0时的KT"), 0, 2)
        self.mooring_kt_j0 = StyledLineEdit()
        self.mooring_kt_j0.setPlaceholderText("从敞水曲线获取")
        input_layout.addWidget(self.mooring_kt_j0, 0, 3)

        input_layout.addWidget(QLabel("J=0时的KQ"), 1, 2)
        self.mooring_kq_j0 = StyledLineEdit()
        self.mooring_kq_j0.setPlaceholderText("从敞水曲线获取")
        input_layout.addWidget(self.mooring_kq_j0, 1, 3)

        # 水的密度
        input_layout.addWidget(QLabel("水的密度 ρ (kg/m³)"), 2, 2)
        self.mooring_rho = StyledLineEdit("1025")
        input_layout.addWidget(self.mooring_rho, 2, 3)

        input_group.setLayout(input_layout)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_fetch_data = StyledButton("获取数据")
        self.btn_fetch_data.clicked.connect(self.fetch_mooring_data)
        btn_layout.addWidget(self.btn_fetch_data)

        self.btn_calc_mooring = StyledButton("系柱计算")
        self.btn_calc_mooring.clicked.connect(self.calculate_mooring)
        btn_layout.addWidget(self.btn_calc_mooring)

        # 结果显示
        result_group = StyledGroupBox("计算结果")
        result_layout = QFormLayout()

        self.mooring_pd = StyledLineEdit()
        self.mooring_pd.setReadOnly(True)
        result_layout.addRow("推功率 P_D (kW):", self.mooring_pd)

        self.mooring_q = StyledLineEdit()
        self.mooring_q.setReadOnly(True)
        result_layout.addRow("转矩 Q (kN·m):", self.mooring_q)

        self.mooring_t = StyledLineEdit()
        self.mooring_t.setReadOnly(True)
        result_layout.addRow("推力 T (kN):", self.mooring_t)

        self.mooring_n_mooring = StyledLineEdit()
        self.mooring_n_mooring.setReadOnly(True)
        result_layout.addRow("系柱转速 N (r/min):", self.mooring_n_mooring)

        result_group.setLayout(result_layout)

        # 添加到主布局
        layout.addWidget(input_group)
        layout.addLayout(btn_layout)
        layout.addWidget(result_group)

        return tab

    def fetch_mooring_data(self):
        """从前面计算获取数据"""
        try:
            # 获取最大航速计算的数据
            if hasattr(self, 'ps_input') and self.ps_input.text():
                self.mooring_ps.setText(self.ps_input.text())
            else:
                self.mooring_ps.setText("6222")

            if hasattr(self, 'n_input') and self.n_input.text():
                self.mooring_n.setText(self.n_input.text())
            else:
                self.mooring_n.setText("155")

            if hasattr(self, 'etas_input') and self.etas_input.text():
                self.mooring_eta_s.setText(self.etas_input.text())
            else:
                self.mooring_eta_s.setText("0.97")

            if hasattr(self, 'etar_input') and self.etar_input.text():
                self.mooring_eta_r.setText(self.etar_input.text())
            else:
                self.mooring_eta_r.setText("1.0")

            # 获取螺旋桨直径从空泡校核结果
            if hasattr(self, 'opt_res') and self.opt_res and 'D' in self.opt_res:
                self.mooring_d.setText(f"{self.opt_res['D']:.4f}")
            else:
                # 如果没有空泡校核结果，尝试从最大航速计算获取
                if hasattr(self, 'tbl_speed') and self.tbl_speed.item(0, 4):
                    try:
                        d_value = float(self.tbl_speed.item(0, 4).text())
                        self.mooring_d.setText(f"{d_value:.4f}")
                    except:
                        self.mooring_d.setText("2.5")

            # 获取KT和KQ在J=0时的值
            blade_num = self.plot_blade_spin.value() if hasattr(self, 'plot_blade_spin') else self.blade_count
            area_ratio = self.plot_area_ratio_spin.value() if hasattr(self, 'plot_area_ratio_spin') else 0.55
            pitch_ratio = self.plot_pitch_ratio_spin.value() if hasattr(self, 'plot_pitch_ratio_spin') else 0.8

            if not self.au_coeffs.update_coefficients_by_blade_count(blade_num):
                QMessageBox.warning(self, "警告", f"暂不支持{blade_num}叶桨的计算")
                return

            # 计算KT和KQ在J=0时的值
            j = 0.0
            kt_j0 = np.float64(self.au_coeffs.current_kt_coeffs[0]['value'])
            for coeff in self.au_coeffs.current_kt_coeffs[1:]:
                term = np.float64(coeff['value'])
                if coeff['i'] > 0:
                    term *= np.power(np.float64(pitch_ratio), np.int32(coeff['i']))
                if coeff['j'] > 0:
                    term *= np.power(np.float64(j), np.int32(coeff['j']))
                if coeff['k'] > 0:
                    term *= np.power(np.float64(area_ratio), np.int32(coeff['k']))
                kt_j0 += term

            ten_kq_j0 = np.float64(self.au_coeffs.current_kq_coeffs[0]['value'])
            for coeff in self.au_coeffs.current_kq_coeffs[1:]:
                term = np.float64(coeff['value'])
                if coeff['i'] > 0:
                    term *= np.power(np.float64(pitch_ratio), np.int32(coeff['i']))
                if coeff['j'] > 0:
                    term *= np.power(np.float64(j), np.int32(coeff['j']))
                if coeff['k'] > 0:
                    term *= np.power(np.float64(area_ratio), np.int32(coeff['k']))
                ten_kq_j0 += term

            kq_j0 = ten_kq_j0 / 10.0

            self.mooring_kt_j0.setText(f"{kt_j0:.6f}")
            self.mooring_kq_j0.setText(f"{kq_j0:.6f}")

            QMessageBox.information(self, "成功", "已获取前面计算的数据")

        except Exception as e:
            QMessageBox.critical(self, "获取数据错误", f"获取数据失败: {str(e)}")

    def calculate_mooring(self):
        """计算系柱工况"""
        try:
            # 安全获取输入值
            ps = self.safe_float_convert(self.mooring_ps.text(), 6222)
            n = self.safe_float_convert(self.mooring_n.text(), 155)
            eta_s = self.safe_float_convert(self.mooring_eta_s.text(), 0.97)
            eta_r = self.safe_float_convert(self.mooring_eta_r.text(), 1.0)
            t0 = self.safe_float_convert(self.mooring_t0.text(), 0.04)
            D = self.safe_float_convert(self.mooring_d.text(), 2.5)
            kt_j0 = self.safe_float_convert(self.mooring_kt_j0.text(), 0.3)
            kq_j0 = self.safe_float_convert(self.mooring_kq_j0.text(), 0.03)
            rho = self.safe_float_convert(self.mooring_rho.text(), 1025)

            # 计算推力
            pd = ps * eta_r * eta_s
            q = pd / (2 * math.pi * n / 60) if n > 0 else 0
            t = (kt_j0 / kq_j0) * (q / D) if (kq_j0 > 0 and D > 0) else 0
            n_mooring = 60 * math.sqrt(t * 1000 / (rho * (D ** 4) * kt_j0)) if (
                    rho > 0 and D > 0 and kt_j0 > 0 and t > 0) else 0

            # 显示结果
            self.mooring_pd.setText(f"{pd:.4f}")
            self.mooring_q.setText(f"{q:.4f}")
            self.mooring_t.setText(f"{t:.4f}")
            self.mooring_n_mooring.setText(f"{n_mooring:.4f}")

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"系柱计算失败: {str(e)}")

    # ===================== 8. 航行特性 =====================
    def create_voyage_characteristics_tab(self):
        """重新设计航行特性计算功能"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 输入参数组
        input_group = StyledGroupBox("航行特性计算参数")
        input_layout = QGridLayout()
        input_layout.setSpacing(6)
        input_layout.setContentsMargins(8, 10, 8, 10)

        # 三个转速输入
        input_layout.addWidget(QLabel("转速1 (r/min)"), 0, 0)
        self.voyage_n1 = StyledLineEdit()
        self.voyage_n1.setPlaceholderText("默认：最大航速转速+10")
        input_layout.addWidget(self.voyage_n1, 0, 1)

        input_layout.addWidget(QLabel("转速2 (r/min)"), 1, 0)
        self.voyage_n2 = StyledLineEdit()
        self.voyage_n2.setPlaceholderText("默认：最大航速转速")
        input_layout.addWidget(self.voyage_n2, 1, 1)

        input_layout.addWidget(QLabel("转速3 (r/min)"), 2, 0)
        self.voyage_n3 = StyledLineEdit()
        self.voyage_n3.setPlaceholderText("默认：最大航速转速-10")
        input_layout.addWidget(self.voyage_n3, 2, 1)

        # 航速范围
        input_layout.addWidget(QLabel("航速范围 (kn)"), 3, 0)
        speed_range_layout = QHBoxLayout()
        self.voyage_v_min = StyledLineEdit("12")
        self.voyage_v_max = StyledLineEdit("17")
        speed_range_layout.addWidget(self.voyage_v_min)
        speed_range_layout.addWidget(QLabel("到"))
        speed_range_layout.addWidget(self.voyage_v_max)
        input_widget = QWidget()
        input_widget.setLayout(speed_range_layout)
        input_layout.addWidget(input_widget, 3, 1)

        # 航速步长
        input_layout.addWidget(QLabel("航速步长"), 4, 0)
        self.voyage_step = StyledLineEdit("1")
        input_layout.addWidget(self.voyage_step, 4, 1)

        # 水的密度
        input_layout.addWidget(QLabel("水的密度 ρ (kg/m³)"), 0, 2)
        self.voyage_rho = StyledLineEdit("1025")
        input_layout.addWidget(self.voyage_rho, 0, 3)

        input_group.setLayout(input_layout)

        # 按钮布局
        btn_layout = QHBoxLayout()
        self.btn_fetch_voyage_data = StyledButton("获取数据")
        self.btn_fetch_voyage_data.clicked.connect(self.fetch_voyage_data)
        btn_layout.addWidget(self.btn_fetch_voyage_data)

        self.btn_calc_voyage = StyledButton("航行计算")
        self.btn_calc_voyage.clicked.connect(self.calculate_voyage_characteristics)
        btn_layout.addWidget(self.btn_calc_voyage)

        self.btn_plot_voyage = StyledButton("绘制曲线")
        self.btn_plot_voyage.clicked.connect(self.plot_voyage_characteristics)
        btn_layout.addWidget(self.btn_plot_voyage)

        # 结果显示
        result_group = StyledGroupBox("计算结果")
        result_layout = QVBoxLayout()

        # 创建表格显示详细计算结果
        self.voyage_table = StyledTableWidget(0, 0)
        result_layout.addWidget(self.voyage_table)

        # 关键点结果显示
        self.voyage_key_results = StyledTextEdit()
        self.voyage_key_results.setReadOnly(True)
        self.voyage_key_results.setMaximumHeight(100)
        result_layout.addWidget(self.voyage_key_results)

        result_group.setLayout(result_layout)

        # 添加到主布局
        layout.addWidget(input_group)
        layout.addLayout(btn_layout)
        layout.addWidget(result_group)

        return tab

    def fetch_voyage_data(self):
        """获取前面计算的数据"""
        try:
            # 获取最大航速计算中的转速
            if hasattr(self, 'n_input') and self.n_input.text():
                base_n = float(self.n_input.text())
                self.voyage_n1.setText(f"{base_n + 10}")
                self.voyage_n2.setText(f"{base_n}")
                self.voyage_n3.setText(f"{base_n - 10}")

            # 获取有效功率曲线数据
            if hasattr(self, 'pe_edit') and self.pe_edit.text():
                pe_data = self.pe_edit.text().split(';')
                if len(pe_data) == 2:
                    speeds = pe_data[0].split(',')
                    if len(speeds) >= 2:
                        self.voyage_v_min.setText(speeds[0].strip())
                        self.voyage_v_max.setText(speeds[-1].strip())

            QMessageBox.information(self, "成功", "已获取前面计算的数据")
        except Exception as e:
            QMessageBox.critical(self, "获取数据错误", f"获取数据失败: {str(e)}")

    def calculate_voyage_characteristics(self):
        """计算航行特性"""
        try:
            # 检查必要的前置计算
            if not self.res or not self.opt_res:
                QMessageBox.warning(self, "警告", "请先完成最大航速和最佳要素确定计算")
                return

            # 获取输入参数
            n1 = self.safe_float_convert(self.voyage_n1.text())
            n2 = self.safe_float_convert(self.voyage_n2.text())
            n3 = self.safe_float_convert(self.voyage_n3.text())
            v_min = self.safe_float_convert(self.voyage_v_min.text(), 12)
            v_max = self.safe_float_convert(self.voyage_v_max.text(), 17)
            step = self.safe_float_convert(self.voyage_step.text(), 1)
            rho = self.safe_float_convert(self.voyage_rho.text(), 1025)

            # 获取螺旋桨参数
            D = self.opt_res['D']
            p_d = self.opt_res['p_d']
            ae_a0 = self.opt_res['AE_A0']
            w = self.res['w']
            t = self.res['t']
            eta_r = self.res['eta_r']
            eta_s = self.res['eta_s']

            # 生成航速序列
            speeds = np.arange(v_min, v_max + step, step)

            # 计算三个转速下的航行特性
            self.voyage_results = {}
            rpm_values = [n1, n2, n3]

            for i, n_rpm in enumerate(rpm_values):
                n_rps = n_rpm / 60.0  # 转换为r/s
                results = []

                for v in speeds:
                    # 计算进速VA
                    VA = 0.5144 * (1 - w) * v  # m/s

                    # 计算进速系数J
                    J = VA / n_rps / D if n_rps > 0 else 0

                    # 计算KT和KQ
                    kt = self.calculate_kt(J, p_d, ae_a0)
                    kq = self.calculate_kq(J, p_d, ae_a0)

                    # 计算推力T (kN)
                    T = kt * rho * (n_rps ** 2) * (D ** 4) / 1000

                    # 计算有效推力功率PTE (kW)
                    PTE = T * (1 - t) * 0.5144 * v

                    # 计算转矩Q (kN·m)
                    Q = kq * rho * (n_rps ** 2) * (D ** 5) / 1000

                    # 计算收到功率PD (kW) - 注意：这里加回了10%功率储备
                    PD = 2 * math.pi * n_rps * Q
                    PD_without_reserve = PD / 0.9  # 去除10%储备

                    # 计算主机功率PS (kW)
                    PS = PD_without_reserve / (eta_r * eta_s)

                    results.append({
                        'V': v, 'VA': VA, 'J': J, 'KT': kt, 'KQ': kq,
                        'T': T, 'PTE': PTE, 'Q': Q, 'PD': PD, 'PS': PS
                    })

                self.voyage_results[f'N={n_rpm}rpm'] = results

            # 获取有效功率曲线数据
            pe_data = self.pe_edit.text().split(';')
            if len(pe_data) == 2:
                pe_speeds = list(map(float, pe_data[0].split(',')))
                pe_powers = list(map(float, pe_data[1].split(',')))
                self.pe_curve = CubicSpline(pe_speeds, pe_powers)
            else:
                # 使用默认有效功率曲线
                pe_speeds = [12, 13, 14, 15, 16, 17]
                pe_powers = [1497, 1953, 2505, 3213, 4070, 5161]
                self.pe_curve = CubicSpline(pe_speeds, pe_powers)

            # 计算三种航行状态的有效功率曲线
            self.voyage_states = {
                'Ⅰ-满载': lambda v: self.pe_curve(v),
                'Ⅱ-压载(85%)': lambda v: 0.85 * self.pe_curve(v),
                'Ⅲ-120%满载': lambda v: 1.2 * self.pe_curve(v)
            }

            # 在表格中显示详细结果
            self.display_voyage_results()

            QMessageBox.information(self, "成功", "航行特性计算完成")

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"航行特性计算失败: {str(e)}")

    def calculate_kt(self, J, p_d, ae_a0):
        """计算推力系数KT"""
        blade_num = self.blade_count
        if not self.au_coeffs.update_coefficients_by_blade_count(blade_num):
            return 0.0

        kt = np.float64(self.au_coeffs.current_kt_coeffs[0]['value'])
        for coeff in self.au_coeffs.current_kt_coeffs[1:]:
            term = np.float64(coeff['value'])
            if coeff['i'] > 0:
                term *= np.power(np.float64(p_d), np.int32(coeff['i']))
            if coeff['j'] > 0:
                term *= np.power(np.float64(J), np.int32(coeff['j']))
            if coeff['k'] > 0:
                term *= np.power(np.float64(ae_a0), np.int32(coeff['k']))
            kt += term

        return max(0.0, kt)

    def calculate_kq(self, J, p_d, ae_a0):
        """计算转矩系数KQ"""
        blade_num = self.blade_count
        if not self.au_coeffs.update_coefficients_by_blade_count(blade_num):
            return 0.0

        ten_kq = np.float64(self.au_coeffs.current_kq_coeffs[0]['value'])
        for coeff in self.au_coeffs.current_kq_coeffs[1:]:
            term = np.float64(coeff['value'])
            if coeff['i'] > 0:
                term *= np.power(np.float64(p_d), np.int32(coeff['i']))
            if coeff['j'] > 0:
                term *= np.power(np.float64(J), np.int32(coeff['j']))
            if coeff['k'] > 0:
                term *= np.power(np.float64(ae_a0), np.int32(coeff['k']))
            ten_kq += term

        kq = ten_kq / 10.0
        return max(0.0, kq)

    def display_voyage_results(self):
        """在表格中显示航行特性计算结果"""
        if not hasattr(self, 'voyage_results'):
            return

        # 获取第一个转速的结果来设置表格
        first_rpm = list(self.voyage_results.keys())[0]
        results = self.voyage_results[first_rpm]

        # 设置表格行数和列数
        num_rows = len(results)
        num_cols = 10  # V, VA, J, KT, KQ, T, PTE, Q, PD, PS

        self.voyage_table.setRowCount(num_rows * 3)  # 三个转速
        self.voyage_table.setColumnCount(num_cols + 1)  # 增加一列显示转速

        # 设置表头
        headers = ["转速", "V (kn)", "VA (m/s)", "J", "KT", "KQ",
                   "T (kN)", "PTE (kW)", "Q (kN·m)", "PD (kW)", "PS (kW)"]
        self.voyage_table.setHorizontalHeaderLabels(headers)

        # 填充表格数据
        row_index = 0
        for rpm_name, results in self.voyage_results.items():
            # 添加转速标题行
            title_item = QTableWidgetItem(rpm_name)
            title_item.setBackground(QColor(200, 220, 240))
            self.voyage_table.setItem(row_index, 0, title_item)
            for col in range(1, num_cols + 1):
                item = QTableWidgetItem("")
                item.setBackground(QColor(200, 220, 240))
                self.voyage_table.setItem(row_index, col, item)
            row_index += 1

            # 填充数据行
            for result in results:
                self.voyage_table.setItem(row_index, 0, QTableWidgetItem(""))
                self.voyage_table.setItem(row_index, 1, QTableWidgetItem(f"{result['V']:.1f}"))
                self.voyage_table.setItem(row_index, 2, QTableWidgetItem(f"{result['VA']:.3f}"))
                self.voyage_table.setItem(row_index, 3, QTableWidgetItem(f"{result['J']:.4f}"))
                self.voyage_table.setItem(row_index, 4, QTableWidgetItem(f"{result['KT']:.4f}"))
                self.voyage_table.setItem(row_index, 5, QTableWidgetItem(f"{result['KQ']:.4f}"))
                self.voyage_table.setItem(row_index, 6, QTableWidgetItem(f"{result['T']:.1f}"))
                self.voyage_table.setItem(row_index, 7, QTableWidgetItem(f"{result['PTE']:.1f}"))
                self.voyage_table.setItem(row_index, 8, QTableWidgetItem(f"{result['Q']:.3f}"))
                self.voyage_table.setItem(row_index, 9, QTableWidgetItem(f"{result['PD']:.1f}"))
                self.voyage_table.setItem(row_index, 10, QTableWidgetItem(f"{result['PS']:.1f}"))
                row_index += 1

    # ===================== 8. 航行特性 =====================
    # ===================== 8. 航行特性 =====================
    def create_voyage_characteristics_tab(self):
        """航行特性计算功能 - 稳定版本"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # 输入参数组
        input_group = StyledGroupBox("航行特性计算参数")
        input_layout = QGridLayout()
        input_layout.setSpacing(6)
        input_layout.setContentsMargins(8, 10, 8, 10)

        # 三个转速输入
        input_layout.addWidget(QLabel("转速1 (r/min)"), 0, 0)
        self.voyage_n1 = StyledLineEdit()
        self.voyage_n1.setPlaceholderText("默认：最大航速转速+10")
        input_layout.addWidget(self.voyage_n1, 0, 1)

        input_layout.addWidget(QLabel("转速2 (r/min)"), 1, 0)
        self.voyage_n2 = StyledLineEdit()
        self.voyage_n2.setPlaceholderText("默认：最大航速转速")
        input_layout.addWidget(self.voyage_n2, 1, 1)

        input_layout.addWidget(QLabel("转速3 (r/min)"), 2, 0)
        self.voyage_n3 = StyledLineEdit()
        self.voyage_n3.setPlaceholderText("默认：最大航速转速-10")
        input_layout.addWidget(self.voyage_n3, 2, 1)

        # 航速范围
        input_layout.addWidget(QLabel("航速范围 (kn)"), 3, 0)
        speed_range_layout = QHBoxLayout()
        self.voyage_v_min = StyledLineEdit("12")
        self.voyage_v_max = StyledLineEdit("17")
        speed_range_layout.addWidget(self.voyage_v_min)
        speed_range_layout.addWidget(QLabel("到"))
        speed_range_layout.addWidget(self.voyage_v_max)
        input_widget = QWidget()
        input_widget.setLayout(speed_range_layout)
        input_layout.addWidget(input_widget, 3, 1)

        # 航速步长
        input_layout.addWidget(QLabel("航速步长"), 4, 0)
        self.voyage_step = StyledLineEdit("1")
        input_layout.addWidget(self.voyage_step, 4, 1)

        # 水的密度
        input_layout.addWidget(QLabel("水的密度 ρ (kg/m³)"), 0, 2)
        self.voyage_rho = StyledLineEdit("1025")
        input_layout.addWidget(self.voyage_rho, 0, 3)

        input_group.setLayout(input_layout)

        # 按钮布局
        btn_layout = QHBoxLayout()
        self.btn_fetch_voyage_data = StyledButton("获取数据")
        self.btn_fetch_voyage_data.clicked.connect(self.fetch_voyage_data)
        btn_layout.addWidget(self.btn_fetch_voyage_data)

        self.btn_calc_voyage = StyledButton("航行计算")
        self.btn_calc_voyage.clicked.connect(self.calculate_voyage_characteristics)
        btn_layout.addWidget(self.btn_calc_voyage)

        self.btn_plot_voyage = StyledButton("绘制曲线")
        self.btn_plot_voyage.clicked.connect(self.plot_voyage_characteristics)
        btn_layout.addWidget(self.btn_plot_voyage)

        # 结果显示
        result_group = StyledGroupBox("计算结果")
        result_layout = QVBoxLayout()

        # 创建表格显示详细计算结果
        self.voyage_table = StyledTableWidget(0, 0)
        result_layout.addWidget(self.voyage_table)

        # 创建关键点显示区域
        self.voyage_keypoints_text = StyledTextEdit()
        self.voyage_keypoints_text.setReadOnly(True)
        self.voyage_keypoints_text.setMaximumHeight(200)
        self.voyage_keypoints_text.setText("关键点数据将在计算后显示...")
        result_layout.addWidget(self.voyage_keypoints_text)

        result_group.setLayout(result_layout)

        # 添加到主布局
        layout.addWidget(input_group)
        layout.addLayout(btn_layout)
        layout.addWidget(result_group)

        # 初始化变量
        self.voyage_results = {}
        self.voyage_intersections = []

        return tab

    def fetch_voyage_data(self):
        """获取前面计算的数据"""
        try:
            # 获取最大航速计算中的转速
            if hasattr(self, 'n_input') and self.n_input.text():
                base_n = float(self.n_input.text())
                self.voyage_n1.setText(f"{base_n + 10}")
                self.voyage_n2.setText(f"{base_n}")
                self.voyage_n3.setText(f"{base_n - 10}")

            # 获取有效功率曲线数据
            if hasattr(self, 'pe_edit') and self.pe_edit.text():
                pe_data = self.pe_edit.text().split(';')
                if len(pe_data) == 2:
                    speeds = pe_data[0].split(',')
                    if len(speeds) >= 2:
                        self.voyage_v_min.setText(speeds[0].strip())
                        self.voyage_v_max.setText(speeds[-1].strip())

            QMessageBox.information(self, "成功", "已获取前面计算的数据")
        except Exception as e:
            QMessageBox.critical(self, "获取数据错误", f"获取数据失败: {str(e)}")

    def calculate_voyage_characteristics(self):
        """计算航行特性"""
        try:
            # 检查必要的前置计算
            if not self.res or not self.opt_res:
                QMessageBox.warning(self, "警告", "请先完成最大航速和最佳要素确定计算")
                return

            # 获取输入参数
            n1 = self.safe_float_convert(self.voyage_n1.text())
            n2 = self.safe_float_convert(self.voyage_n2.text())
            n3 = self.safe_float_convert(self.voyage_n3.text())
            v_min = self.safe_float_convert(self.voyage_v_min.text(), 12)
            v_max = self.safe_float_convert(self.voyage_v_max.text(), 17)
            step = self.safe_float_convert(self.voyage_step.text(), 1)
            rho = self.safe_float_convert(self.voyage_rho.text(), 1025)

            # 验证航速范围
            if v_min >= v_max:
                QMessageBox.warning(self, "输入错误", "航速最小值必须小于最大值")
                return

            # 获取螺旋桨参数
            D = self.opt_res['D']
            p_d = self.opt_res['p_d']
            ae_a0 = self.opt_res['AE_A0']
            w = self.res['w']
            t = self.res['t']
            eta_r = self.res['eta_r']
            eta_s = self.res['eta_s']

            # 生成航速序列
            speeds = np.arange(v_min, v_max + step, step)
            speeds = speeds[(speeds >= v_min) & (speeds <= v_max)]

            if len(speeds) == 0:
                QMessageBox.warning(self, "输入错误", "航速范围内无有效数据点")
                return

            # 计算三个转速下的航行特性
            self.voyage_results = {}
            rpm_values = [n1, n2, n3]

            for i, n_rpm in enumerate(rpm_values):
                n_rps = n_rpm / 60.0
                results = []

                for v in speeds:
                    VA = 0.5144 * (1 - w) * v
                    J = VA / (n_rps * D) if (n_rps > 0 and D > 0) else 0
                    J = max(0.0, min(1.5, J))

                    kt = self.calculate_kt(J, p_d, ae_a0)
                    kq = self.calculate_kq(J, p_d, ae_a0)

                    T = kt * rho * (n_rps ** 2) * (D ** 4) / 1000
                    PTE = T * (1 - t) * 0.5144 * v
                    Q = kq * rho * (n_rps ** 2) * (D ** 5) / 1000
                    PD = 2 * math.pi * n_rps * Q
                    PD_without_reserve = PD / 0.9
                    PS = PD_without_reserve / (eta_r * eta_s)

                    results.append({
                        'V': v, 'VA': VA, 'J': J, 'KT': kt, 'KQ': kq,
                        'T': T, 'PTE': PTE, 'Q': Q, 'PD': PD, 'PS': PS
                    })

                self.voyage_results[f'N={n_rpm}rpm'] = results

            # 获取有效功率曲线数据
            pe_data = self.pe_edit.text().split(';')
            if len(pe_data) == 2:
                pe_speeds = list(map(float, pe_data[0].split(',')))
                pe_powers = list(map(float, pe_data[1].split(',')))
                self.pe_curve = CubicSpline(pe_speeds, pe_powers)
            else:
                pe_speeds = np.linspace(v_min, v_max, 6)
                pe_powers = np.linspace(1000, 5000, 6)
                self.pe_curve = CubicSpline(pe_speeds, pe_powers)

            # 三种航行状态
            self.voyage_states = {
                'Ⅰ-满载': lambda v: self.pe_curve(v),
                'Ⅱ-压载(85%)': lambda v: 0.85 * self.pe_curve(v),
                'Ⅲ-120%满载': lambda v: 1.2 * self.pe_curve(v)
            }

            # 在表格中显示详细结果
            self.display_voyage_results()

            QMessageBox.information(self, "成功", "航行特性计算完成")

        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"航行特性计算失败: {str(e)}")

    def plot_voyage_characteristics(self):
        """绘制航行特性图，只标记交点圆点"""
        if not hasattr(self, 'voyage_results') or not self.voyage_results:
            QMessageBox.warning(self, "警告", "请先完成航行特性计算")
            return

        try:
            # 创建绘图窗口
            self.voyage_plot_window = QDialog(self)
            self.voyage_plot_window.setWindowTitle("航行特性图")
            self.voyage_plot_window.setGeometry(100, 100, 1000, 800)

            # 创建图表
            fig = Figure(figsize=(10, 8), dpi=100)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self.voyage_plot_window)

            plt.rcParams.update({'font.size': 10})

            # 创建包含第一象限和第四象限的图表
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])

            # 获取航速范围
            first_rpm = list(self.voyage_results.keys())[0]
            speeds = [result['V'] for result in self.voyage_results[first_rpm]]
            v_min, v_max = min(speeds), max(speeds)
            v_fine = np.linspace(v_min, v_max, 200)

            # 颜色和样式
            colors = ['red', 'blue', 'green']
            line_styles = ['-', '--', '-.']
            markers = ['o', 's', '^']

            # 存储所有交点信息
            intersection_points = []

            # 第一象限：绘制有效功率曲线和PTE曲线
            # 绘制三种航行状态的有效功率曲线
            for i, (state_name, pe_func) in enumerate(self.voyage_states.items()):
                pe_values = [pe_func(v) for v in v_fine]
                ax1.plot(v_fine, pe_values, color=colors[i], linestyle=line_styles[i],
                         linewidth=2, label=state_name)

            # 绘制三个转速的PTE曲线并计算交点
            for i, (rpm_name, results) in enumerate(self.voyage_results.items()):
                # 使用更精确的插值计算PTE曲线
                pte_speeds = [result['V'] for result in results]
                pte_values = [result['PTE'] for result in results]

                # 使用三次样条插值获得平滑的PTE曲线
                pte_spline = CubicSpline(pte_speeds, pte_values)
                pte_smooth = pte_spline(v_fine)

                ax1.plot(v_fine, pte_smooth, color=colors[i], linestyle=line_styles[i % len(line_styles)],
                         linewidth=2, label=f'{rpm_name} PTE')

                # 计算该转速PTE曲线与所有状态PE曲线的交点
                for j, (state_name, pe_func) in enumerate(self.voyage_states.items()):
                    # 计算差值函数
                    diff_func = lambda v: pte_spline(v) - pe_func(v)

                    # 在航速范围内寻找交点
                    intersections = []
                    for k in range(len(v_fine) - 1):
                        v1, v2 = v_fine[k], v_fine[k + 1]
                        diff1, diff2 = diff_func(v1), diff_func(v2)

                        # 检查是否有交点
                        if diff1 * diff2 <= 0:
                            try:
                                # 使用二分法精确求解交点
                                v_intersect = fsolve(diff_func, (v1 + v2) / 2)[0]

                                # 确保交点在有效范围内
                                if v_min <= v_intersect <= v_max:
                                    pte_intersect = pte_spline(v_intersect)
                                    pe_intersect = pe_func(v_intersect)

                                    # 找到对应的PS值
                                    ps_intersect = None
                                    for result in results:
                                        if abs(result['V'] - v_intersect) < 0.1:
                                            ps_intersect = result['PS']
                                            break

                                    if ps_intersect is None:
                                        # 使用插值计算PS
                                        ps_speeds = [r['V'] for r in results]
                                        ps_values = [r['PS'] for r in results]
                                        ps_spline = CubicSpline(ps_speeds, ps_values)
                                        ps_intersect = ps_spline(v_intersect)

                                    intersection_info = {
                                        'rpm': rpm_name,
                                        'state': state_name,
                                        'speed': v_intersect,
                                        'pte': pte_intersect,
                                        'pe': pe_intersect,
                                        'ps': ps_intersect,
                                        'color': colors[j]
                                    }

                                    # 检查是否已经存在相似的交点
                                    is_duplicate = False
                                    for existing in intersections:
                                        if (abs(existing['speed'] - v_intersect) < 0.5 and
                                                existing['state'] == state_name):
                                            is_duplicate = True
                                            break

                                    if not is_duplicate:
                                        intersections.append(intersection_info)
                            except:
                                continue

                    # 存储所有交点
                    intersection_points.extend(intersections)

            # 在图表上标记所有交点（只保留圆点，不添加标注）
            for point in intersection_points:
                ax1.plot(point['speed'], point['pte'], 'o',
                         color=point['color'], markersize=8, zorder=5)

            ax1.set_ylabel('功率 PE, PTE (kW)', fontsize=12)
            ax1.set_title('第一象限: 有效功率和推力功率曲线', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', fontsize=10)

            # 第四象限：绘制Ps曲线
            for i, (rpm_name, results) in enumerate(self.voyage_results.items()):
                ps_values = [result['PS'] for result in results]
                ax2.plot(speeds, ps_values, color=colors[i], linestyle=line_styles[i % len(line_styles)],
                         linewidth=2, label=f'{rpm_name} PS', marker=markers[i % len(markers)], markersize=4)

            ax2.set_xlabel('航速 V (kn)', fontsize=12)
            ax2.set_ylabel('主机功率 PS (kW)', fontsize=12)
            ax2.set_title('第四象限: 主机功率曲线', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best', fontsize=10)

            # 存储交点信息供后续使用
            self.voyage_intersections = intersection_points

            # 更新关键点显示
            self.update_keypoints_display()

            fig.tight_layout()

            # 添加到布局
            layout = QVBoxLayout()
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            self.voyage_plot_window.setLayout(layout)
            self.voyage_plot_window.show()

        except Exception as e:
            QMessageBox.critical(self, "绘图错误", f"绘制航行特性图时发生错误: {str(e)}")

    def display_voyage_results(self):
        """在表格中显示航行特性计算结果"""
        if not hasattr(self, 'voyage_results'):
            return

        # 获取第一个转速的结果来设置表格
        first_rpm = list(self.voyage_results.keys())[0]
        results = self.voyage_results[first_rpm]

        # 设置表格行数和列数
        num_rows = len(results)
        num_cols = 10

        # 设置总行数：详细结果（三个转速）
        total_rows = num_rows * 3

        self.voyage_table.setRowCount(total_rows)
        self.voyage_table.setColumnCount(num_cols + 1)

        # 设置表头
        headers = ["转速", "V (kn)", "VA (m/s)", "J", "KT", "KQ",
                   "T (kN)", "PTE (kW)", "Q (kN·m)", "PD (kW)", "PS (kW)"]
        self.voyage_table.setHorizontalHeaderLabels(headers)

        # 填充详细计算结果
        row_index = 0
        for rpm_name, results in self.voyage_results.items():
            # 添加转速标题行
            title_item = QTableWidgetItem(rpm_name)
            title_item.setBackground(QColor(200, 220, 240))
            self.voyage_table.setItem(row_index, 0, title_item)
            for col in range(1, num_cols + 1):
                item = QTableWidgetItem("")
                item.setBackground(QColor(200, 220, 240))
                self.voyage_table.setItem(row_index, col, item)
            row_index += 1

            # 填充数据行
            for result in results:
                self.voyage_table.setItem(row_index, 0, QTableWidgetItem(""))
                self.voyage_table.setItem(row_index, 1, QTableWidgetItem(f"{result['V']:.1f}"))
                self.voyage_table.setItem(row_index, 2, QTableWidgetItem(f"{result['VA']:.3f}"))
                self.voyage_table.setItem(row_index, 3, QTableWidgetItem(f"{result['J']:.4f}"))
                self.voyage_table.setItem(row_index, 4, QTableWidgetItem(f"{result['KT']:.4f}"))
                self.voyage_table.setItem(row_index, 5, QTableWidgetItem(f"{result['KQ']:.4f}"))
                self.voyage_table.setItem(row_index, 6, QTableWidgetItem(f"{result['T']:.1f}"))
                self.voyage_table.setItem(row_index, 7, QTableWidgetItem(f"{result['PTE']:.1f}"))
                self.voyage_table.setItem(row_index, 8, QTableWidgetItem(f"{result['Q']:.3f}"))
                self.voyage_table.setItem(row_index, 9, QTableWidgetItem(f"{result['PD']:.1f}"))
                self.voyage_table.setItem(row_index, 10, QTableWidgetItem(f"{result['PS']:.1f}"))
                row_index += 1

    def update_keypoints_display(self):
        """更新关键点显示"""
        if not hasattr(self, 'voyage_intersections') or not self.voyage_intersections:
            self.voyage_keypoints_text.setText("未找到关键点数据")
            return

        # 生成关键点文本
        keypoints_text = "关键点数据 (PTE与PE曲线交点):\n\n"

        # 按转速分组
        rpm_groups = {}
        for point in self.voyage_intersections:
            rpm = point['rpm']
            if rpm not in rpm_groups:
                rpm_groups[rpm] = []
            rpm_groups[rpm].append(point)

        # 生成关键点文本
        for rpm, points in rpm_groups.items():
            keypoints_text += f"{rpm}:\n"
            for point in points:
                keypoints_text += (f"  • {point['state']}: 航速 {point['speed']:.2f} kn, "
                                   f"PTE = {point['pte']:.1f} kW, "
                                   f"主机功率 = {point['ps']:.1f} kW\n")
            keypoints_text += "\n"

        # 添加最佳性能点
        if self.voyage_intersections:
            best_point = max(self.voyage_intersections, key=lambda x: x['speed'])
            keypoints_text += f"最佳性能点:\n"
            keypoints_text += (f"  • {best_point['rpm']} - {best_point['state']}: "
                               f"航速 {best_point['speed']:.2f} kn, "
                               f"PTE = {best_point['pte']:.1f} kW, "
                               f"主机功率 = {best_point['ps']:.1f} kW\n")

        # 更新文本显示
        self.voyage_keypoints_text.setText(keypoints_text)
    # ---------- 工具函数 ----------
    def clear_all(self):
        """清空所有数据"""
        try:
            # 清空输入框 - 只清空存在的属性
            input_attributes = ['ps_input', 'n_input', 'etas_input', 'etar_input', 'w_input', 't_input', 'vs_input',
                                'pe_edit']

            for attr in input_attributes:
                if hasattr(self, attr):
                    widget = getattr(self, attr)
                    if hasattr(widget, 'clear'):
                        widget.clear()

            # 清空表格 - 只清空存在的属性
            table_attributes = ['tbl_speed', 'tbl_strength', 'cavitation_table', 'tbl_mass_results', 'tbl_mass_details']

            for attr in table_attributes:
                if hasattr(self, attr):
                    widget = getattr(self, attr)
                    if hasattr(widget, 'clearContents'):
                        widget.clearContents()

            # 清空文本显示 - 只清空存在的属性
            text_attributes = ['result_text', 'txt_pc_result', 'voyage_key_results']

            for attr in text_attributes:
                if hasattr(self, attr):
                    widget = getattr(self, attr)
                    if hasattr(widget, 'clear'):
                        widget.clear()

            # 重置变量
            self.res = {}
            self.opt_res = {}
            self.mass_details = []
            self.cavitation_results = {}
            self.optimum_results = {}

            QMessageBox.information(self, "成功", "所有数据已清空")
        except Exception as e:
            QMessageBox.critical(self, "清空错误", f"清空数据时发生错误: {str(e)}")


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置高质量字体渲染
    font = QFont()
    font.setFamily("Microsoft YaHei, SimSun")  # 优先使用微软雅黑，其次宋体
    font.setPointSize(9)
    font.setWeight(QFont.Normal)
    font.setStyleStrategy(QFont.PreferAntialias)  # 启用抗锯齿

    app.setFont(font)

    # 设置应用调色板以获得一致的颜色
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(236, 240, 241))
    palette.setColor(QPalette.WindowText, QColor(44, 62, 80))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(248, 249, 250))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(44, 62, 80))
    palette.setColor(QPalette.Text, QColor(44, 62, 80))
    palette.setColor(QPalette.Button, QColor(52, 152, 219))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    # 设置高DPI属性
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    window = PropellerDesignSystem()
    window.show()
    sys.exit(app.exec_())