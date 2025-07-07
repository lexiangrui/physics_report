import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
import math
import matplotlib as mpl
import platform
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def setup_matplotlib_font():
    """配置matplotlib字体设置"""
    system = platform.system()
    font_config = {
        'Darwin': ['Arial Unicode MS', 'SimHei'],  # macOS
        'Windows': ['SimHei', 'Microsoft YaHei'],  # Windows
        'Linux': ['WenQuanYi Micro Hei', 'SimHei']  # Linux
    }
    mpl.rcParams['font.sans-serif'] = font_config.get(system, ['SimHei'])
    mpl.rcParams['axes.unicode_minus'] = False

setup_matplotlib_font()

class BaseAnalyzer:
    """分析器基类，提供共用的基础功能"""
    
    def __init__(self, data_file):
        """初始化基础分析器"""
        self.data_file = data_file
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """加载并预处理数据"""
        try:
            self.data = pd.read_csv(self.data_file)
            if not all(col in self.data.columns for col in ['time', 'x_position']):
                raise ValueError("缺少必要的列：'time' 或 'x_position'")
            self.data = self.data.fillna(0)
            print("数据加载成功！")
        except Exception as e:
            print(f"数据加载失败: {e}")
            self.data = None
    
    def _smooth_data(self, data, window_length=15):
        """平滑数据处理"""
        try:
            window_length = min(window_length, len(data)-1)
            window_length = window_length - 1 if window_length % 2 == 0 else window_length
            polyorder = min(3, window_length-1)
            return savgol_filter(data, window_length, polyorder)
        except Exception as e:
            print(f"平滑数据失败: {e}")
            return data

class PendulumAnalyzer(BaseAnalyzer):
    """分析单摆运动的类"""
    
    # 标准重力加速度（武汉地区）
    STANDARD_GRAVITY = 9.7936  # m/s²
    
    def __init__(self, data_file, pendulum_length=None):
        """
        初始化分析器
        :param data_file: 轨迹数据CSV文件路径
        :param pendulum_length: 摆长（米）
        """
        super().__init__(data_file)
        self.pendulum_length = pendulum_length
        self.period = None

    @staticmethod
    def harmonic_function(t, A, omega, phi, C):
        """简谐运动函数模型"""
        return A * np.sin(omega * t + phi) + C

    @staticmethod
    def damped_harmonic_function(t, A, omega, phi, gamma, C):
        """
        阻尼简谐运动函数模型
        θ(t) = Ae^(-γt)sin(ωt + φ) + C
        其中γ = β/m为衰减系数，β为阻尼系数(N·s/m)，m为质量(kg)
        γ的单位为s⁻¹，直接决定振幅衰减速率
        """
        return A * np.exp(-gamma * t) * np.sin(omega * t + phi) + C

    @staticmethod
    def nonlinear_period(theta_max):
        """计算大摆角修正系数"""
        return 1 + (1/16) * (theta_max**2) + (11/3072) * (theta_max**4)

    def _analyze_fft(self, times, positions):
        """FFT分析"""
        dt = np.mean(np.diff(times))
        n = len(positions)
        n_padded = 8 * n
        
        signal = positions - np.mean(positions)
        window = np.hanning(n)
        padded_signal = np.zeros(n_padded)
        padded_signal[:n] = signal * window
        
        fft_values = fft(padded_signal)
        freqs = fftfreq(n_padded, dt)
        pos_mask = freqs > 0
        
        freq_range = (freqs >= 0.1) & (freqs <= 5.0) & pos_mask
        valid_freqs = freqs[freq_range]
        valid_magnitudes = np.abs(fft_values)[freq_range]
        
        if len(valid_magnitudes) == 0:
            return 0, None, None
            
        main_idx = np.argmax(valid_magnitudes)
        main_freq = valid_freqs[main_idx]
        
        return 1.0/main_freq, valid_freqs, valid_magnitudes

    def _analyze_peaks(self, times, positions):
        """峰值检测分析"""
        min_distance = int(0.5 / np.mean(np.diff(times)))
        prominence = 0.1 * np.std(positions)
        
        peaks, _ = find_peaks(positions, 
                             distance=min_distance,
                             prominence=prominence,
                             width=2)
        
        if len(peaks) < 2:
            return 0, 0, peaks
            
        peak_times = times[peaks]
        periods = np.diff(peak_times)
        mean_period = np.mean(periods)
        std_period = np.std(periods)
        
        return mean_period, std_period, peaks

    def _curve_fit_analysis(self, times, positions, initial_period=None):
        """曲线拟合分析"""
        if len(times) < 5:
            return 0
            
        try:
            A0 = (np.max(positions) - np.min(positions)) / 2
            omega0 = 2 * np.pi / (initial_period or (times[-1] - times[0]) / 5)
            
            params, _ = curve_fit(
                self.harmonic_function,
                times,
                positions,
                p0=[A0, omega0, 0, np.mean(positions)],
                bounds=([0, 2*np.pi/5, -np.pi, -np.inf],
                       [np.inf, 2*np.pi/0.5, np.pi, np.inf])
            )
            
            return 2 * np.pi / params[1]
        except Exception as e:
            print(f"曲线拟合失败: {e}")
            return 0

    def analyze_period(self, theta_max_degrees=None):
        """分析运动周期，返回分析结果字典"""
        if self.data is None:
            return None

        times = self.data['time'].values
        positions = self._smooth_data(self.data['x_position'].values)
        
        fft_period, freqs, magnitudes = self._analyze_fft(times, positions)
        peak_period, peak_std, peaks = self._analyze_peaks(times, positions)
        fit_period = self._curve_fit_analysis(times, positions, fft_period)
        
        periods, methods, errors = [], [], []
        corrected_periods, corrected_g_values, corrected_errors = [], [], []
        
        if fft_period > 0:
            periods.append(fft_period)
            methods.append("FFT")
            
        if peak_period > 0:
            periods.append(peak_period)
            methods.append("Peak")
            
        if fit_period > 0:
            periods.append(fit_period)
            methods.append("Fit")
        
        min_error_period, min_error, min_error_method = 0, float('inf'), ""
        min_error_corrected_period, min_error_corrected_g = 0, 0
        
        if self.pendulum_length:
            for i, period in enumerate(periods):
                g = self.calculate_gravity(period)
                if g:
                    error = abs(g - self.STANDARD_GRAVITY) / self.STANDARD_GRAVITY * 100
                    errors.append(error)
                    
                    if theta_max_degrees is not None:
                        corrected_g = self.calculate_gravity(period, theta_max_degrees)
                        corrected_error = abs(corrected_g - self.STANDARD_GRAVITY) / self.STANDARD_GRAVITY * 100
                        
                        corrected_periods.append(period)
                        corrected_g_values.append(corrected_g)
                        corrected_errors.append(corrected_error)
                        
                        if corrected_error < min_error:
                            min_error = corrected_error
                            min_error_period = period
                            min_error_method = methods[i]
                            min_error_corrected_g = corrected_g
                    else:
                        if error < min_error:
                            min_error = error
                            min_error_period = period
                            min_error_method = methods[i]
                            min_error_corrected_g = g
                else:
                    errors.append(float('inf'))
                    if theta_max_degrees is not None:
                        corrected_periods.append(period)
                        corrected_g_values.append(None)
                        corrected_errors.append(float('inf'))
        
        self.period = min_error_period if min_error_period > 0 else (periods[0] if periods else 0)
            
        return {
            'fft_period': fft_period,
            'peak_period': peak_period,
            'peak_std': peak_std,
            'fit_period': fit_period,
            'min_error_period': min_error_period,
            'min_error': min_error,
            'min_error_method': min_error_method,
            'min_error_corrected_g': min_error_corrected_g,
            'peaks': peaks,
            'freqs': freqs,
            'magnitudes': magnitudes,
            'methods': methods,
            'periods': periods,
            'errors': errors,
            'corrected_periods': corrected_periods,
            'corrected_g_values': corrected_g_values,
            'corrected_errors': corrected_errors,
            'theta_max_degrees': theta_max_degrees
        }

    def calculate_gravity(self, period, theta_max_degrees=None):
        """计算重力加速度"""
        if not all([self.pendulum_length, period > 0]):
            return None
            
        correction = 1.0
        if theta_max_degrees is not None:
            correction = self.nonlinear_period(math.radians(theta_max_degrees))
            
        return 4 * (math.pi ** 2) * self.pendulum_length / ((period ** 2) / (correction ** 2))

    def plot_analysis(self, theta_max_degrees=None):
        """绘制分析图表，返回 matplotlib Figure 对象"""
        if self.data is None:
            return None

        try:
            fig = Figure(figsize=(12, 10))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            
            results = self.analyze_period(theta_max_degrees)
            
            times = self.data['time'].values
            positions = self._smooth_data(self.data['x_position'].values)
            
            self._plot_position_time(ax1, times, positions, results)
            
            self._plot_spectrum(ax2, results)
            
            fig.tight_layout() 
            
            return fig 
            
        except Exception as e:
            print(f"绘制分析图表时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_position_time(self, ax, times, positions, results):
        """绘制位置-时间图"""
        ax.plot(times, self.data['x_position'].values, 'b.', label='原始位置', alpha=0.3)
        ax.plot(times, positions, 'r-', label='平滑后位置', linewidth=2)
        
        dt = np.mean(np.diff(times))
        sample_info = (f"采样信息:\n"
                     f"数据点数: {len(times)}\n"
                     f"采样率: {1/dt:.2f} Hz\n"
                     f"奈奎斯特频率: {1/(2*dt):.2f} Hz")
        ax.annotate(sample_info, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        if results['peaks'] is not None:
            peak_times = times[results['peaks']]
            peak_positions = positions[results['peaks']]
            ax.plot(peak_times, peak_positions, 'go', label='峰值点', markersize=6)
            
            if results['fit_period'] > 0:
                self._plot_fit_curve(ax, times, positions, results)

        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('X 方向位置 (像素)')
        ax.set_title('小球X方向位置随时间变化')
        ax.legend(loc='upper right')
        ax.grid(True)

    def _plot_fit_curve(self, ax, times, positions, results):
        """绘制拟合曲线"""
        try:
            A0 = (np.max(positions) - np.min(positions)) / 2
            omega = 2 * np.pi / results['fit_period']
            params, _ = curve_fit(
                self.harmonic_function, 
                times, 
                positions, 
                p0=[A0, omega, 0, np.mean(positions)],
                maxfev=5000
            )
            
            t_fit = np.linspace(times[0], times[-1], 1000)
            y_fit = self.harmonic_function(t_fit, *params)
            ax.plot(t_fit, y_fit, 'g--', linewidth=1, label='拟合曲线')
            
            fit_info = (f"拟合信息:\n"
                      f"周期: {2*np.pi/params[1]:.3f}s\n"
                      f"振幅: {abs(params[0]):.1f}\n"
                      f"相位: {params[2]:.2f}")
            ax.annotate(fit_info, xy=(0.02, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except Exception as e:
            print(f"绘制拟合曲线失败: {e}")

    def _plot_spectrum(self, ax, results):
        """绘制频谱图"""
        if results['freqs'] is not None and results['magnitudes'] is not None:
            freq_mask = (results['freqs'] >= 0.1) & (results['freqs'] <= 5.0)
            plot_freqs = results['freqs'][freq_mask]
            plot_mags = results['magnitudes'][freq_mask]
            
            if len(plot_freqs) > 0:
                ax.plot(plot_freqs, plot_mags)
                
                main_idx = np.argmax(plot_mags)
                main_freq = plot_freqs[main_idx]
                main_mag = plot_mags[main_idx]
                
                ax.plot(main_freq, main_mag, 'ro')
                ax.annotate(
                    f"主频率: {main_freq:.3f} Hz\n"
                    f"周期: {1/main_freq:.3f}s", 
                    xy=(main_freq, main_mag),
                    xytext=(main_freq*1.1, main_mag*0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05)
                )
                
                spectrum_info = (f"频谱分析:\n"
                               f"主频率: {main_freq:.3f} Hz\n"
                               f"对应周期: {1/main_freq:.3f}s")
                ax.annotate(spectrum_info, xy=(0.02, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                ax.set_xlim(0.1, 5.0)
                ax.set_ylim(0, max(plot_mags) * 1.2)
        
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('幅度')
        ax.set_title('频谱分析')
        ax.grid(True)

    def print_results(self, theta_max_degrees=None):
        """返回分析结果字符串"""
        results = self.analyze_period(theta_max_degrees)
        if not results:
            return "分析失败，无结果。"
            
        output = ["\n=== 分析结果 ==="]
        output.append(f"使用的测量方法: {', '.join(results['methods'])}")
        
        output.append("\n未修正的周期结果:")
        for i, (method, period) in enumerate(zip(results['methods'], results['periods'])):
            if method == 'Peak':
                output.append(f"峰值检测周期: {period:.3f} ± {results['peak_std']:.3f} 秒")
            else:
                output.append(f"{method}分析周期: {period:.3f} 秒")
            
            if self.pendulum_length and i < len(results['errors']):
                g = self.calculate_gravity(period)
                output.append(f"  重力加速度: {g:.3f} m/s²")
                output.append(f"  相对误差: {results['errors'][i]:.2f}%")
        
        if theta_max_degrees is not None and self.pendulum_length:
            correction = self.nonlinear_period(math.radians(theta_max_degrees))
            output.append(f"\n最大摆角: {theta_max_degrees:.1f}°")
            output.append(f"大摆角修正系数: {correction:.6f}")
            
            output.append("\n应用大摆角修正后的结果:")
            for i, (method, period) in enumerate(zip(results['methods'], results['periods'])):
                if i < len(results['corrected_periods']):
                    corrected_g = results['corrected_g_values'][i]
                    corrected_error = results['corrected_errors'][i]
                    
                    if corrected_g:
                        output.append(f"{method}分析:")
                        output.append(f"  修正后重力加速度: {corrected_g:.3f} m/s²")
                        output.append(f"  修正后相对误差: {corrected_error:.2f}%")
        
        if results['min_error_period'] > 0:
            output.append(f"\n相对误差最小的结果:")
            output.append(f"方法: {results['min_error_method']}")
            output.append(f"周期: {results['min_error_period']:.3f} 秒")
            output.append(f"相对误差: {results['min_error']:.2f}%")
            output.append(f"注: 最终采用此结果作为周期值")
        else:
             output.append(f"注: 最终采用{results['methods'][0]}方法结果作为周期值")
        
        if self.pendulum_length:
            output.append(f"\n摆长: {self.pendulum_length:.3f} 米")
            
            if results['min_error_corrected_g']:
                final_method = results['min_error_method']
                final_g = results['min_error_corrected_g']
                final_error = results['min_error']
                
                output.append(f"\n最终重力加速度估计 ({final_method}):")
                output.append(f"g = {final_g:.3f} m/s² (误差: {final_error:.2f}%)")
                output.append(f"标准值 (武汉): {self.STANDARD_GRAVITY} m/s²")
            else:
                output.append("\n无法计算重力加速度。")
                
        return "\n".join(output)

class ViscosityAnalyzer(BaseAnalyzer):
    """分析振动阻尼系数的类"""
    
    def __init__(self, data_file, mass=1.0):
        """初始化阻尼分析器"""
        super().__init__(data_file)
        self.damping_coef = None  
        self.natural_frequency = None  
        self.mass = mass  
    
    @staticmethod
    def damped_harmonic_function(t, A, omega, phi, gamma, C):
        """
        阻尼简谐运动函数模型
        θ(t) = Ae^(-γt)sin(ωt + φ) + C
        其中γ = β/m为衰减系数，β为阻尼系数(N·s/m)，m为质量(kg)
        γ的单位为s⁻¹，直接决定振幅衰减速率
        """
        return A * np.exp(-gamma * t) * np.sin(omega * t + phi) + C
    
    def calculate_damping_coefficient(self, peaks_data):
        """
        使用对数衰减法计算阻尼系数
        β = (m/nT)ln(θ₀/θₙ)，其中：
        m: 摆球质量 (kg)
        n: 周期数
        T: 周期 (s)
        θ₀: 初始振幅
        θₙ: n个周期后的振幅
        
        阻尼系数β的单位为 N·s/m (牛顿·秒/米)
        衰减系数γ = β/m，单位为 s⁻¹
        """
        if len(peaks_data['times']) < 2:
            return None, None
            
        periods = np.diff(peaks_data['times'])
        T = np.mean(periods)
        
        theta_0 = np.abs(peaks_data['amplitudes'][0])
        theta_n = np.abs(peaks_data['amplitudes'][-1])
        n = len(periods)  
        
        beta = (self.mass / (n * T)) * np.log(theta_0 / theta_n)
        
        return beta, T
    
    def analyze_damping(self):
        """分析阻尼系数"""
        if self.data is None:
            print("无数据可分析")
            return None
        
        times = self.data['time'].values
        positions = self._smooth_data(self.data['x_position'].values)
        
        try:
            peaks, _ = find_peaks(positions, prominence=0.1*np.std(positions))
            if len(peaks) < 2:
                print("未能找到足够的峰值点进行分析")
                return None
                
            peak_times = times[peaks]
            peak_positions = positions[peaks]
            mean_position = np.mean(positions)
            peak_amplitudes = peak_positions - mean_position
            
            peaks_data = {
                'times': peak_times,
                'amplitudes': peak_amplitudes
            }
            
            beta, T = self.calculate_damping_coefficient(peaks_data)
            if beta is None:
                return None
                
            omega_0 = 2 * np.pi / T
            self.natural_frequency = omega_0
            
            self.damping_coef = beta
            
            A0 = np.abs(peak_amplitudes[0])
            gamma = beta / self.mass
            params, _ = curve_fit(
                self.damped_harmonic_function,
                times,
                positions,
                p0=[A0, omega_0, 0, gamma, mean_position],
                bounds=([0, 2*np.pi/10, -np.pi, 0, -np.inf],
                       [np.inf, 2*np.pi/0.1, np.pi, 2.0, np.inf]),
                maxfev=10000
            )
            
            return {
                'amplitude': params[0],
                'angular_frequency': params[1],
                'phase': params[2],
                'damping_coefficient': beta,
                'offset': params[4],
                'period': T,
                'peaks_data': peaks_data
            }
            
        except Exception as e:
            print(f"分析阻尼运动失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_damped_analysis(self):
        """绘制阻尼分析图表，返回 matplotlib Figure 对象"""
        if self.data is None:
            return None
        
        try:
            fig = Figure(figsize=(12, 10))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            
            times = self.data['time'].values
            positions = self._smooth_data(self.data['x_position'].values)
            results = self.analyze_damping()
            
            if not results:
                return None
            
            self._plot_damped_position(ax1, times, positions, results)
            
            self._plot_log_decay(ax2, times, positions, results)
            
            fig.tight_layout()
            
            return fig 
            
        except Exception as e:
            print(f"绘制阻尼分析图表时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_damped_position(self, ax, times, positions, results):
        """绘制阻尼位置图"""
        ax.plot(times, self.data['x_position'].values, 'b.', label='原始位置', alpha=0.3)
        ax.plot(times, positions, 'r-', label='平滑后位置', linewidth=2)
        
        gamma = results['damping_coefficient'] / self.mass
        
        t_fit = np.linspace(times[0], times[-1], 1000)
        y_fit = self.damped_harmonic_function(
            t_fit, 
            results['amplitude'], 
            results['angular_frequency'], 
            results['phase'], 
            gamma,  
            results['offset']
        )
            
        ax.plot(t_fit, y_fit, 'g--', linewidth=1, label='阻尼拟合曲线')
        
        fit_info = (f"拟合信息:\n"
                   f"周期: {results['period']:.3f}s\n"
                   f"阻尼系数 β: {results['damping_coefficient']:.6f} N·s/m\n"
                   f"初始振幅: {results['amplitude']:.1f} 像素")
        ax.annotate(fit_info, xy=(0.02, 0.05), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        envelope_upper = results['amplitude'] * np.exp(-gamma * t_fit) + results['offset']
        envelope_lower = -results['amplitude'] * np.exp(-gamma * t_fit) + results['offset']
            
        ax.plot(t_fit, envelope_upper, 'k--', alpha=0.5)
        ax.plot(t_fit, envelope_lower, 'k--', alpha=0.5)
        
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('X 方向位置 (像素)')
        ax.set_title('小球阻尼振动分析')
        ax.legend(loc='upper right')
        ax.grid(True)

    def _plot_log_decay(self, ax, times, positions, results):
        """绘制对数衰减图"""
        peaks_data = results['peaks_data']
        if len(peaks_data['times']) > 1:
            log_amplitudes = np.log(np.abs(peaks_data['amplitudes']))
            
            ax.plot(peaks_data['times'], log_amplitudes, 'ro-', label='测量值')
            
            gamma = results['damping_coefficient'] / self.mass
            
            t_fit = np.linspace(peaks_data['times'][0], peaks_data['times'][-1], 100)
            log_fit = np.log(np.abs(results['amplitude'])) - gamma * t_fit
            ax.plot(t_fit, log_fit, 'b-', label='理论衰减线')
            
            n = len(peaks_data['times']) - 1
            theta_0 = np.abs(peaks_data['amplitudes'][0])
            theta_n = np.abs(peaks_data['amplitudes'][-1])
            
            damp_info = (f"阻尼分析:\n"
                        f"摆球质量 m: {self.mass:.3f} kg\n"
                        f"初始振幅 θ₀: {theta_0:.2f}\n"
                        f"n个周期后振幅 θ': {theta_n:.2f}\n"
                        f"阻尼系数 β: {results['damping_coefficient']:.6f} N·s/m\n"
            )
            ax.annotate(damp_info, xy=(0.02, 0.05), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="orange", alpha=0.8))
        
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('振幅对数值 ln(θ)')
        ax.set_title('对数衰减分析')
        ax.grid(True)
        ax.legend()
    
    def print_results(self):
        """返回阻尼分析结果字符串"""
        results = self.analyze_damping()
        if not results:
            return "阻尼分析失败，无结果。"
            
        output = ["\n=== 阻尼振动分析结果 ==="]
        
        peaks_data = results['peaks_data']
        n = len(peaks_data['times']) - 1
        theta_0 = np.abs(peaks_data['amplitudes'][0])
        theta_n = np.abs(peaks_data['amplitudes'][-1])
        
        gamma = results['damping_coefficient'] / self.mass
        
        output.append(f"摆球质量 m: {self.mass:.3f} kg")
        output.append(f"初始振幅 θ₀: {theta_0:.2f}")
        output.append(f"n个周期后振幅 θ': {theta_n:.2f}")
        output.append(f"周期数 n: {n}")
        output.append(f"周期 T: {results['period']:.3f} 秒")
        output.append(f"阻尼系数 β: {results['damping_coefficient']:.6f} N·s/m")
        output.append(f"衰减系数 γ = β/m: {gamma:.6f} s⁻¹")
        
        if self.natural_frequency:
             output.append(f"固有频率 ω₀: {self.natural_frequency:.4f} rad/s")
             if results['damping_coefficient'] > 1e-9: 
                 Q = self.mass * self.natural_frequency / results['damping_coefficient']
                 output.append(f"品质因数 Q: {Q:.2f}")
             else:
                 output.append(f"品质因数 Q: N/A (阻尼系数过小)")
        else:
             output.append("未能计算固有频率。")
             output.append("品质因数 Q: N/A")


        if gamma > 1e-9: 
            tau = 1 / gamma
            output.append(f"衰减时间常数 τ: {tau:.2f} 秒")
        else:
            output.append(f"衰减时间常数 τ: N/A (衰减系数过小)")

        return "\n".join(output)
