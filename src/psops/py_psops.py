####################################################################################################
# The project Py_PSOPS is the python API for a Power System Electromechanical Simulator called PSOPS.
# PSOPS stands for Power System Optimal Power Parameter Selection.
# The related paper' preprint can be found on arxiv, http://arxiv.org/abs/2110.00931.
# All the techniques used in PSOPS have been published, references' DOIs are as follows
# [1] https://ieeexplore.ieee.org/document/8798601/.
# [2] https://ieeexplore.ieee.org/document/8765766/.
# [3] https://ieeexplore.ieee.org/document/8283798/.
####################################################################################################
from ctypes import *
import platform
import os
import numpy as np
import datetime
import pathlib
from scipy.stats import qmc
from tqdm import tqdm
# import ray
# import torch

# Environment variable setting
parent_dir, _ = os.path.split(os.path.abspath(__file__))

array_1d_double = np.ctypeslib.ndpointer(
    dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

if platform.system() == 'Windows':
    os.environ['path'] += ';%s\\dll_win' % parent_dir
elif platform.system() == 'Linux':
    os.environ['PATH'] += ';%s/dll_linux' % parent_dir
else:
    print('Unknown operating system. Please check!')
    exit(-1)


class Py_PSOPS:
    def __init__(self, flg=0, rng=None):
        """Construction.

        Args:
            flg (int, optional): An internal flag used to distinguish the API in parallel computing. Defaults to 0.
            rng (numpy.random.Generator, optional): The random seed (random generator since numpy 1.19). Defaults to None. If none, np.random.default_rng() will be used.
        """        
        # api flag
        self.__flg = flg
        # random state
        self.set_random_state(rng=rng)
        # working direction
        self.__workingDir = parent_dir
        # dll path
        dll_path = self.__workingDir
        # load .dll or .so
        if platform.system() == 'Windows':
            dll_path += '\\dll_win\\PSOPS_Source.dll'
        elif platform.system() == 'Linux':
            # dll_path += '/dll_linux/libPSOPS-Console-QT-V.so.1.0.0'
            dll_path += '/dll_linux/libPSOPS_Source.so.1.0.0'
        else:
            print('Unknown operating system. Please check!')
            exit(-1)
        self._load_dll(dll_path)
        # load config file
        self._load_configuration(self.__workingDir + '/config.txt')
        # psops function test
        self._basic_fun_test()
        # basic info
        self._get_basic_info()
        # create buffer
        self._create_buffer()
        # get initial state
        self._get_initial_state()
        # time stamp
        self.__timeStamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        # current total_step
        self.__cur_total_step = -1
        # bounds
        self.__generator_p_bounds = [0, 1.5]
        self.__load_pq_bounds = [0.7, 1.2]
        print('api for psops creation successful.')

    def __del__(self):
        """Deconstruction
        """        
        print('api for psops deletion successful.')

    def set_random_state(self, rng):
        if type(rng) is int: 
            assert rng >= 0, f'int rng should not be negative, current rng is {rng}.'
            self.__rng = np.random.default_rng(rng)
        elif rng is not None:
            self.__rng = rng
        else:
            self.__rng = np.random.default_rng()

    def _load_dll(self, dll_path: str):
        """Function of loading the .dll or .so file of PSOPS with ctypes library. Load in all the external functions of PSOPS. 
           Most of the details can be found in the corresponding python functions. 

        Args:
            dll_path (str): The directory of the .dll or .so file.
        """        
        # paths = os.environ['path']
        # path_list = paths.split(";")
        # for path in path_list:
        #     if path != "." and pathlib.Path(path).exists():
        #         os.add_dll_directory(path)
        self.__psDLL = cdll.LoadLibrary(dll_path)
        # mpi settings. TODO deal with mpi
        # mpi initiation
        self.__psDLL.init_MPI.argtypes = None
        self.__psDLL.init_MPI.restype = c_bool
        # process mapping, bind the processes to certain CPU cores/processors.
        self.__psDLL.mapping_MPI.argtypes = None
        self.__psDLL.mapping_MPI.restype = c_bool
        # finalize mpi
        self.__psDLL.finalize_MPI.argtypes = None
        self.__psDLL.finalize_MPI.restype = c_bool
        # basic fun. TODO change config by python
        # read settings from the config file locating at './config.txt'
        self.__psDLL.read_Settings.argtypes = [c_wchar_p, c_int]
        self.__psDLL.read_Settings.restype = c_bool
        # call the calculation function
        self.__psDLL.cal_Functions.argtypes = [c_int]
        self.__psDLL.cal_Functions.restype = c_bool
        # calculation
        # power flow
        self.__psDLL.cal_Power_Flow_Basic_Newton_Raphson.argtypes = None
        self.__psDLL.cal_Power_Flow_Basic_Newton_Raphson.restype = c_int
        # stability simulation
        self.__psDLL.cal_Transient_Stability_Simulation_TI_SV.argtypes = [c_double, c_int]
        self.__psDLL.cal_Transient_Stability_Simulation_TI_SV.restype = c_int
        # cal info and cal control
        self.__psDLL.get_Info_LF_Iter.argtypes = [c_int]
        self.__psDLL.get_Info_LF_Iter.restype = c_int
        self.__psDLL.get_Info_TE.argtypes = None
        self.__psDLL.get_Info_TE.restype = c_double
        self.__psDLL.get_Info_DT.argtypes = None
        self.__psDLL.get_Info_DT.restype = c_double
        self.__psDLL.get_Info_Max_Step.argtypes = None
        self.__psDLL.get_Info_Max_Step.restype = c_int
        self.__psDLL.get_Info_Finish_Step.argtypes = None
        self.__psDLL.get_Info_Finish_Step.restype = c_int
        self.__psDLL.get_Info_Fault_Step_Sequence.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Info_Fault_Step_Sequence.restype = c_bool
        self.__psDLL.get_Info_Fault_Time_Sequence.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Info_Fault_Time_Sequence.restype = c_bool
        self.__psDLL.set_Info_TS_Step_Network_State.argtypes = [c_int, c_bool, c_int]
        self.__psDLL.set_Info_TS_Step_Network_State.restype = c_bool
        self.__psDLL.set_Info_TS_Step_Element_State.argtypes = [c_int, c_bool, c_int]
        self.__psDLL.set_Info_TS_Step_Element_State.restype = c_bool
        self.__psDLL.set_Info_TS_Step_All_State.argtypes = [c_int, c_bool, c_int]
        self.__psDLL.set_Info_TS_Step_All_State.restype = c_bool
        # asynchronous systems
        self.__psDLL.get_ACSystem_Number.argtypes = None 
        self.__psDLL.get_ACSystem_Number.restype = c_int
        self.__psDLL.get_ACSystem_TS_CurStep_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_ACSystem_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_ACSystem_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_ACSystem_TS_All_Result.restype = c_bool
        # bus
        self.__psDLL.get_Bus_Number.argtypes = [c_int]
        self.__psDLL.get_Bus_Number.restype = c_int
        self.__psDLL.get_Bus_Name.argtypes = [c_int, c_int]
        self.__psDLL.get_Bus_Name.restype = c_char_p
        self.__psDLL.get_Bus_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Bus_Sys_No.restype = c_int
        self.__psDLL.get_Bus_VMax.argtypes = [c_int, c_int]
        self.__psDLL.get_Bus_VMax.restype = c_double
        self.__psDLL.get_Bus_VMin.argtypes = [c_int, c_int]
        self.__psDLL.get_Bus_VMin.restype = c_double
        self.__psDLL.get_Bus_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Bus_LF_Result.restype = c_bool
        self.__psDLL.get_Bus_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Bus_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Bus_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Bus_TS_All_Result.restype = c_bool
        # acline
        self.__psDLL.get_ACLine_Number.argtypes = [c_int]
        self.__psDLL.get_ACLine_Number.restype = c_int
        self.__psDLL.get_ACLine_I_No.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_I_No.restype = c_int
        self.__psDLL.get_ACLine_J_No.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_J_No.restype = c_int
        self.__psDLL.get_ACLine_Sys_No.argtypes = [c_int]
        self.__psDLL.get_ACLine_Sys_No.restype = c_int
        self.__psDLL.get_ACLine_No.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_No.restype = c_long
        self.__psDLL.get_ACLine_Current_Capacity.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_Current_Capacity.restype = c_double
        self.__psDLL.get_ACLine_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_ACLine_LF_Result.restype = c_bool
        self.__psDLL.get_ACLine_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_ACLine_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_ACLine_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_ACLine_TS_All_Result.restype = c_bool
        # transformer
        self.__psDLL.get_Transformer_Number.argtypes = [c_int]
        self.__psDLL.get_Transformer_Number.restype = c_int
        self.__psDLL.get_Transformer_I_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_I_No.restype = c_int
        self.__psDLL.get_Transformer_J_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_J_No.restype = c_int
        self.__psDLL.get_Transformer_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Transformer_Sys_No.restype = c_int
        self.__psDLL.get_Transformer_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_No.restype = c_long
        self.__psDLL.get_Transformer_Current_Capacity.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_Current_Capacity.restype = c_double
        self.__psDLL.get_Transformer_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Transformer_LF_Result.restype = c_bool
        self.__psDLL.get_Transformer_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Transformer_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Transformer_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Transformer_TS_All_Result.restype = c_bool
        # generator
        self.__psDLL.get_Generator_Number.argtypes = [c_int]
        self.__psDLL.get_Generator_Number.restype = c_int
        self.__psDLL.get_Generator_Bus_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Bus_No.restype = c_int
        self.__psDLL.get_Generator_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Generator_Sys_No.restype = c_int
        self.__psDLL.get_Generator_LF_Bus_Type.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_LF_Bus_Type.restype = c_int
        self.__psDLL.get_Generator_V0.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_V0.restype = c_double
        self.__psDLL.set_Generator_V0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Generator_V0.restype = c_bool
        self.__psDLL.get_Generator_P0.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_P0.restype = c_double
        self.__psDLL.set_Generator_P0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Generator_P0.restype = c_bool
        self.__psDLL.get_Generator_PMax.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_PMax.restype = c_double
        self.__psDLL.get_Generator_PMin.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_PMin.restype = c_double
        self.__psDLL.get_Generator_QMax.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_QMax.restype = c_double
        self.__psDLL.get_Generator_QMin.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_QMin.restype = c_double
        self.__psDLL.get_Generator_Tj.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Tj.restype = c_double
        self.__psDLL.get_Generator_TS_Type.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_TS_Type.restype = c_char_p
        self.__psDLL.set_Generator_Environment_Status.argtypes = [c_int, c_double, c_int, c_int]
        self.__psDLL.set_Generator_Environment_Status.restype = c_bool
        self.__psDLL.get_Generator_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_LF_Result.restype = c_bool
        self.__psDLL.get_Generator_TS_Result_Dimension.argtypes = [c_int, c_int, c_bool]
        self.__psDLL.get_Generator_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int, c_bool]
        self.__psDLL.get_Generator_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Generator_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Generator_TS_All_Result.restype = c_bool
        # exciter
        self.__psDLL.get_Generator_Exciter_TS_Result_Dimension.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Exciter_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_Exciter_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_Exciter_TS_CurStep_Result.restype = c_bool
        # governor
        self.__psDLL.get_Generator_Governor_TS_Result_Dimension.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Governor_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_Governor_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_Governor_TS_CurStep_Result.restype = c_bool
        # pss
        self.__psDLL.get_Generator_PSS_TS_Result_Dimension.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_PSS_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_PSS_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_PSS_TS_CurStep_Result.restype = c_bool
        # load
        self.__psDLL.get_Load_Number.argtypes = [c_int]
        self.__psDLL.get_Load_Number.restype = c_int
        self.__psDLL.get_Load_Bus_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_Bus_No.restype = c_int
        self.__psDLL.get_Load_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Load_Sys_No.restype = c_int
        self.__psDLL.get_Load_P0.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_P0.restype = c_double
        self.__psDLL.set_Load_P0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Load_P0.restype = c_bool
        self.__psDLL.get_Load_Q0.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_Q0.restype = c_double
        self.__psDLL.set_Load_Q0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Load_Q0.restype = c_bool
        self.__psDLL.get_Load_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Load_LF_Result.restype = c_bool
        self.__psDLL.get_Load_TS_Result_Dimension.argtypes = [c_int, c_int, c_bool]
        self.__psDLL.get_Load_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Load_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int, c_bool]
        self.__psDLL.get_Load_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Load_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Load_TS_All_Result.restype = c_bool
        # network
        self.__psDLL.get_Network_N_Non_Zero_Element.argtypes = [c_int]
        self.__psDLL.get_Network_N_Non_Zero_Element.restype = c_int
        self.__psDLL.get_Network_N_Inverse_Non_Zero_Element.argtypes = [c_int]
        self.__psDLL.get_Network_N_Inverse_Non_Zero_Element.restype = c_int
        self.__psDLL.get_Network_N_ACSystem_Check_Connectivity.argtypes = [c_int]
        self.__psDLL.get_Network_N_ACSystem_Check_Connectivity.restype = c_int
        self.__psDLL.get_Network_Bus_Connectivity_Flag.argtypes = [c_int, c_int]
        self.__psDLL.get_Network_Bus_Connectivity_Flag.restype = c_int
        self.__psDLL.get_Network_ACLine_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Network_ACLine_Connectivity.restype = c_bool
        self.__psDLL.set_Network_ACLine_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Network_ACLine_Connectivity.restype = c_bool
        self.__psDLL.get_Network_Transformer_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Network_Transformer_Connectivity.restype = c_bool
        self.__psDLL.set_Network_Transformer_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Network_Transformer_Connectivity.restype = c_bool
        self.__psDLL.set_Network_Rebuild_All_Network_Data.argtypes = None
        self.__psDLL.set_Network_Rebuild_All_Network_Data.restype = c_bool
        self.__psDLL.get_Network_Generator_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Network_Generator_Connectivity.restype = c_bool
        self.__psDLL.set_Network_Generator_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Network_Generator_Connectivity.restype = c_bool
        self.__psDLL.get_Network_Load_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Network_Load_Connectivity.restype = c_bool
        self.__psDLL.set_Network_Load_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Network_Load_Connectivity.restype = c_bool
        self.__psDLL.get_Network_Admittance_Matrix_Full.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Network_Admittance_Matrix_Full.restype = c_bool
        self.__psDLL.get_Network_Impedence_Matrix_Full.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Network_Impedence_Matrix_Full.restype = c_bool
        self.__psDLL.get_Network_Impedence_Matrix_Factorized.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Network_Impedence_Matrix_Factorized.restype = c_bool
        # fault and disturbance
        self.__psDLL.set_Fault_Disturbance_Clear_All.argtypes = None
        self.__psDLL.set_Fault_Disturbance_Clear_All.restype = c_bool
        self.__psDLL.set_Fault_Disturbance_Add_Fault.argtypes = [c_int, c_double, c_double, c_double, c_int, c_int, c_bool, c_int]
        self.__psDLL.set_Fault_Disturbance_Add_Fault.restype = c_bool
        self.__psDLL.set_Fault_Disturbance_Add_Disturbance.argtypes = [c_int, c_double, c_double, c_int, c_int, c_bool, c_int]
        self.__psDLL.set_Fault_Disturbance_Add_Disturbance.restype = c_bool

    def _load_configuration(self, cfg_path: str):
        """Load configuration file.

        Args:
            cfg_path (str): The directory of the config file.
        """        
        self.__config_path = cfg_path
        cfg = open(cfg_path, "r")
        for line in cfg.readlines():
            # get the directory of the original data file.
            if line[0:3].lower() == 'dir': 
                self.__fullFilePath = self.__workingDir + line[4:].strip()[1:]
                (self.__absFilePath, temp_file_name) = os.path.split(self.__fullFilePath)
                (self.__absFileName, extension) = os.path.splitext(temp_file_name)
                print(self.__absFilePath, self.__absFileName, extension)
        cfg.close()

    def _basic_fun_test(self):
        """Basic function test before utilization.
        """        
        assert self.__psDLL.read_Settings(self.__config_path, len(self.__config_path)), 'read settings failure!'
        assert self.__psDLL.cal_Functions(1), 'basic function check failure!'

    def _get_basic_info(self):
        """Get basic information of the power system
        """     
        # total number of asynchronous ac systems   
        self.__nACSystem = self.__psDLL.get_ACSystem_Number()
        # total number of buses
        self.__nBus = self.__psDLL.get_Bus_Number(-1)
        assert self.__nBus >= 0, 'system total bus number wrong, please check!'
        # build an numpy array for all the bus names
        self.__allBusName = list()
        for bus_no in range(self.__nBus):
            tmp = self.__psDLL.get_Bus_Name(bus_no, -1)
            assert tmp is not None,  "bus name is empty, please check bus no.!"
            self.__allBusName.append(string_at(tmp, -1).decode('gbk'))
        self.__allBusName = np.array(self.__allBusName)
        # total number of aclines
        self.__nACLine = self.__psDLL.get_ACLine_Number(-1)
        assert self.__nACLine >= 0, 'system number wrong, please check!'
        # total number of transformers
        self.__nTransformer = self.__psDLL.get_Transformer_Number(-1)
        assert self.__nTransformer >= 0, 'total number of transformer wrong, please check!'
        # total number of generators
        self.__nGenerator = self.__psDLL.get_Generator_Number(-1)
        assert self.__nGenerator >= 0, 'total number of generator wrong, please check!'
        # total number of loads
        self.__nLoad = self.__psDLL.get_Load_Number(-1)
        assert self.__nLoad >= 0, 'total number of load wrong, please check!'
        # total number of non-zero elements in the factor table
        self.__nNonzero = self.__psDLL.get_Network_N_Non_Zero_Element(-1)
        assert self.__nNonzero >= 0, 'total number of non-zero element wrong, please check!'
        # total number of non-zero elements in the inverse factor table
        self.__nInverseNonZero = self.__psDLL.get_Network_N_Inverse_Non_Zero_Element(-1)
        assert self.__nInverseNonZero >= 0, 'total number of inverse non-zeror wrong, please check!'

    def _create_buffer(self):
        """Create buffers for loading data from PSOPS.
        """        
        self.__bufferSize = max(
            self.__nBus * 6 * (self.get_info_ts_max_step() + 500), max(self.__nNonzero, self.__nInverseNonZero + 100) * 6)
        self.__doubleBuffer = np.zeros(self.__bufferSize, np.float64)
        self.__intBuffer = np.zeros(self.__bufferSize, np.int32)
        self.__boolBuffer = np.zeros(self.__bufferSize, bool)
        print("Buffer Created in Python.")
        # run buffer tests
        # self._buffer_tests()

    def _get_initial_state(self):
        """Get initial state of the power system.
        """        
        # gen lf bus type
        bus_type = self.get_generator_all_lf_bus_type()
        self.__indexSlack = np.arange(self.__nGenerator, dtype=np.int32)[bus_type == 'slack']
        self.__indexCtrlGen = np.arange(self.__nGenerator, dtype=np.int32)[bus_type != 'slack']
        # gen v set
        self.__generator_v_origin = self.get_generator_all_v_set()
        # gen p set
        self.__generator_p_origin = self.get_generator_all_p_set()
        # load p set
        self.__load_p_origin = self.get_load_all_p_set()
        # load q set
        self.__load_q_origin = self.get_load_all_q_set()

    ################################################################################################
    # Power Flow
    ################################################################################################
    def cal_power_flow_basic_nr(self):
        """Power flow: calculation using Newton-Raphson method.

        Returns:
            int: The number of iterations of the power flow solution.
        """        
        return self.__psDLL.cal_Power_Flow_Basic_Newton_Raphson()
    
    def get_power_flow_bounds(self, 
                              generator_v_list=None,
                              generator_p_list=None,
                              load_p_list=None,
                              load_q_list=None,
                              load_max=None,
                              load_min=None,
                              sys_no=-1):
        """power flow: get power flow bounds according to settings.

        Args:
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            load_max (float, optional): the load upper bound settings. Defaults to None. 
                None means loading the default settings shown by self.__load_pq_bounds.
                -1 means loading the current load settings.
                Others mean loading the original load settings and multiply the given load_max.
            load_min (float, optional): the load lower bound settings. Similar to load_max. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list: [upper_bound, lower_bound]. The two bounds meet with np.array([vg, pg, pl, ql]).
        """        
        # gen v max&min
        if generator_v_list is None: generator_v_list = np.arange(self.get_generator_number(sys_no))
        gen_vmax = self.get_generator_all_vmax(generator_v_list, sys_no)
        gen_vmin = self.get_generator_all_vmin(generator_v_list, sys_no)
        # gen p max&min
        if generator_p_list is None: generator_p_list = self.get_generator_all_ctrl(None, sys_no)
        gen_pmax = self.get_generator_all_pmax(generator_p_list, sys_no)
        gen_pmin = self.get_generator_all_pmin(generator_p_list, sys_no)
        # gen p check
        if np.all(gen_pmax == 0) and np.all(gen_pmin == 0):
            if sys_no != -1:
                for i in range(sys_no):
                    generator_p_list += self.get_bus_number(i)
            gen_pmax = self.__generator_p_origin[generator_p_list] * self.__generator_p_bounds[1]
            gen_pmin = self.__generator_p_origin[generator_p_list] * self.__generator_p_bounds[0]
        # load bound
        load_max = self.__load_pq_bounds[1] if load_max is None else load_max
        load_min = self.__load_pq_bounds[0] if load_min is None else load_min
        # load p max&min
        if load_p_list is None: load_p_list = np.arange(self.get_load_number(sys_no))
        if load_max == -1: load_pmax = self.get_load_all_p_set(load_p_list, sys_no)
        if load_min == -1: load_pmin = self.get_load_all_p_set(load_p_list, sys_no)
        if sys_no != -1:
            for i in range(sys_no): load_p_list += self.get_bus_number(i)
        if load_max != -1: load_pmax = self.__load_p_origin[load_p_list] * load_max
        if load_min != -1: load_pmin = self.__load_p_origin[load_p_list] * load_min
        # load q max&min
        if load_q_list is None: load_q_list = np.arange(self.get_load_number(sys_no))
        if load_max == -1: load_qmax = self.get_load_all_q_set(load_q_list, sys_no)
        if load_min == -1: load_qmin = self.get_load_all_q_set(load_q_list, sys_no)
        if sys_no != -1:
            for i in range(sys_no): load_q_list += self.get_bus_number(i)
        if load_max != -1: load_qmax = self.__load_q_origin[load_q_list] * load_max
        if load_min != -1: load_qmin = self.__load_q_origin[load_q_list] * load_min
        # concatenate bounds
        lower = np.concatenate((gen_vmin, gen_pmin, load_pmin, load_qmin))
        upper = np.concatenate((gen_vmax, gen_pmax, load_pmax, load_qmax))
        idx = lower > upper
        lower[idx], upper[idx] = upper[idx], lower[idx]
        assert np.any(lower > upper) == False, "get lf bounds failed, please check!"
        return [lower, upper]

    def set_power_flow_initiation(self, 
                                  sample: np.ndarray, 
                                  generator_v_list=None,
                                  generator_p_list=None,
                                  load_p_list=None,
                                  load_q_list=None,
                                  sys_no=-1):
        """Power flow: set power flow initiation state.

        Args:
            sample (np.ndarray): the sample setting, np.array([vg, pg, pl, ql]).
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        sample_part = sample.copy()
        # gen v set
        if generator_v_list is None:
            generator_v_list = np.arange(self.get_generator_number(sys_no))
        self.set_generator_all_v_set(sample_part[:len(generator_v_list)], generator_v_list, sys_no)
        sample_part = sample_part[len(generator_v_list):]
        # gen p set
        if generator_p_list is None:
            generator_p_list = self.get_generator_all_ctrl(None, sys_no)
        self.set_generator_all_p_set(sample_part[:len(generator_p_list)], generator_p_list, sys_no)
        sample_part = sample_part[len(generator_p_list):]
        # load p set
        if load_p_list is None:
            load_p_list = np.arange(self.get_load_number(sys_no))
        self.set_load_all_p_set(sample_part[:len(load_p_list)], load_p_list, sys_no)
        sample_part = sample_part[len(load_p_list):]
        # load q set
        if load_q_list is None:
            load_q_list = np.arange(self.get_load_number(sys_no))
        self.set_load_all_q_set(sample_part[:len(load_q_list)], load_q_list, sys_no)

    def get_power_flow_original_status(self):
        """Power flow: get the original power flow state.

        Returns:
            ndarray float: the power flow settings are np.array([vg, pg, pl, ql]).
        """        
        return np.concatenate((self.__generator_v_origin, self.__generator_p_origin[self.__indexCtrlGen], self.__load_p_origin, self.__load_q_origin))

    def set_power_flow_original_status(self):
        """Power flow: resume power flow to the original state.
        """        
        self.set_power_flow_initiation(self.get_power_flow_original_status())

    def get_power_flow_status_check(self):
        """Power flow: get power flow state. [convergence, slack generation is bigger than 0, all the voltages are within acceptable range]

        Returns:
            list: power flow state, [convergence, slack, voltage].
        """        
        [converge, slack, voltage] = [False, False, False]
        if self.cal_power_flow_basic_nr() > 0:
            converge = True
            slack_p = self.get_generator_all_lf_result(self.__indexSlack)[:, 0]
            slack_p_max = self.get_generator_all_pmax(self.__indexSlack)
            slack_p_min = self.get_generator_all_pmin(self.__indexSlack)
            if np.all(slack_p >= slack_p_min) and np.all(slack_p <= slack_p_max): slack = True
            lf_v = self.get_bus_all_lf_result()[:, 0]
            if np.all(lf_v < self.get_bus_all_vmax()) and np.all(lf_v > self.get_bus_all_vmin()):
                voltage = True
        return [converge, slack, voltage]

    # pf sampler, gen_v, ctrl_gen_p, load_p, load_q
    def get_power_flow_sample_simple_random(
        self, 
        num=1, 
        generator_v_list=None, 
        generator_p_list=None, 
        load_p_list=None, 
        load_q_list=None, 
        load_max=None, 
        load_min=None, 
        sys_no=-1, 
        check_converge=True, 
        check_slack=True, 
        check_voltage=True,
        termination_num=None,
        need_print=False
        ):
        """Power flow: get power flow samples using simple random sampling method.

        Args:
            num (int, optional): number of samples. Defaults to 1.
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            load_max (float, optional): the load upper bound settings. Defaults to None. 
                None means loading the default settings shown by self.__load_pq_bounds.
                -1 means loading the current load settings.
                Others mean loading the original load settings and multiply the given load_max.
            load_min (float, optional): the load lower bound settings. Similar to load_max. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            check_converge (bool, optional): flag shows whether checking convergence. Defaults to True.
            check_slack (bool, optional): flag shows whether checking active power generation of slack generators. Defaults to True.
            check_voltage (bool, optional): flag shows whether checking nodal voltages. Defaults to True.
            termination_num (int, optional): if the number of samples is too much, the sampling process will terminate. Default to None.
            need_print (bool, optional): flag shows whether the total number of sampled pf will be printed. Defaults to False.

        Returns:
            ndarray: power flow settings.
        """        
        [lower_bounds, upper_bounds] = self.get_power_flow_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        bound_size = len(lower_bounds)
        sample_buffer = list()
        iter_buffer = list()
        sample_total = 0
        sample_no = 0
        for _ in range(num):
            [converge, valid_slack, valid_v] = [False, False, False]
            while (False in [converge, valid_slack, valid_v]):
                r_vector = self.__rng.random(bound_size)
                cur_status = lower_bounds + (upper_bounds - lower_bounds) * r_vector
                sample_total += 1
                self.set_power_flow_initiation(cur_status, generator_v_list, generator_p_list, load_p_list, load_q_list)
                # total_load = sum(self.get_load_all_p_set())
                # total_slack = sum(self.get_generator_all_pmax(self.__indexSlack))
                # total_ctrl = sum(self.get_generator_all_p_set(self.__indexCtrlGen))
                # if total_load < total_ctrl or total_load > total_ctrl + total_slack:
                #     continue
                [converge, valid_slack, valid_v] = self.get_power_flow_status_check()
                [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
                iter_buffer.append(self.get_info_lf_iter())
                if False not in [converge, valid_slack, valid_v]:
                    break
                if termination_num is not None and sample_total >= termination_num: break
            if termination_num is not None and sample_total >= termination_num: break
            sample_buffer.append(cur_status)
            sample_no += 1
        if need_print: print(f'total sample: {sample_total}, valid sample: {sample_no}')  
        return sample_buffer
    
    # pf sampler, gen_v, ctrl_gen_p, load_p, load_q
    def get_power_flow_sample_lhb(
        self, 
        num=1, 
        generator_v_list=None, 
        generator_p_list=None, 
        load_p_list=None, 
        load_q_list=None, 
        load_max=None, 
        load_min=None, 
        sys_no=-1, 
        check_converge=True, 
        check_slack=True, 
        check_voltage=True, 
        seed=None,
        termination_num=None,
        need_print=False
    ):
        """Power flow: get power flow samples using the latin hypercube method.

        Args:
            num (int, optional): number of samples. Defaults to 1.
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            load_max (float, optional): the load upper bound settings. Defaults to None. 
                None means loading the default settings shown by self.__load_pq_bounds.
                -1 means loading the current load settings.
                Others mean loading the original load settings and multiply the given load_max.
            load_min (float, optional): the load lower bound settings. Similar to load_max. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            check_converge (bool, optional): flag shows whether checking convergence. Defaults to True.
            check_slack (bool, optional): flag shows whether checking active power generation of slack generators. Defaults to True.
            check_voltage (bool, optional): flag shows whether checking nodal voltages. Defaults to True.
            seed (int/rng, optional): seed of lhb sampler, int or rng.
            termination_num (int, optional): if the number of samples is too much, the sampling process will terminate. Default to None.
            need_print (bool, optional): flag shows whether the total number of sampled pf will be printed. Defaults to False.

        Returns:
            ndarray: power flow settings.
        """        
        [lower_bounds, upper_bounds] = self.get_power_flow_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        sample_buffer = list()
        iter_buffer = list()
        if seed is None: lhb_sampler = qmc.LatinHypercube(d=lower_bounds.shape[0])
        else: lhb_sampler = qmc.LatinHypercube(d=lower_bounds.shape[0], seed=seed)
        n_valid_sample = 0
        sample_total = 0
        while (n_valid_sample < num):
            norm_sample = lhb_sampler.random(n=num)
            norm_sample = qmc.scale(norm_sample, lower_bounds, upper_bounds)
            for cur_status in norm_sample:
                # cur_status = lower_bounds + (upper_bounds - lower_bounds) * r_vector
                sample_total += 1
                self.set_power_flow_initiation(cur_status, generator_v_list, generator_p_list, load_p_list, load_q_list)
                [converge, valid_slack, valid_v] = self.get_power_flow_status_check()
                [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
                iter_buffer.append(self.get_info_lf_iter())
                if False not in [converge, valid_slack, valid_v]:
                    n_valid_sample += 1
                    sample_buffer.append(cur_status)
                    if n_valid_sample >= num:
                        break
                if termination_num is not None and sample_total >= termination_num: break
            if termination_num is not None and sample_total >= termination_num: break
        if need_print: print(f'total sample: {sample_total}, valid sample: {n_valid_sample}')  
        return sample_buffer
        
    def get_power_flow_sample_stepwise(
        self, 
        num=1, 
        generator_v_list=None, 
        generator_p_list=None, 
        load_p_list=None, 
        load_q_list=None, 
        load_max=None, 
        load_min=None, 
        sys_no=-1, 
        check_converge=True, 
        check_slack=True, 
        check_voltage=True,
        termination_num=None,
        need_print=False
    ):
        """Power flow: get power flow samples using stepwise sampling method.
           1. sampling load p. 
           2. sampling generator p. 5 times.
           3. sampling load q. 5 times.
           4. sampling voltages. 5 times.

        Args:
            num (int, optional): number of samples. Defaults to 1.
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            load_max (float, optional): the load upper bound settings. Defaults to None. 
                None means loading the default settings shown by self.__load_pq_bounds.
                -1 means loading the current load settings.
                Others mean loading the original load settings and multiply the given load_max.
            load_min (float, optional): the load lower bound settings. Similar to load_max. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            check_converge (bool, optional): flag shows whether checking convergence. Defaults to True.
            check_slack (bool, optional): flag shows whether checking active power generation of slack generators. Defaults to True.
            check_voltage (bool, optional): flag shows whether checking nodal voltages. Defaults to True.
            termination_num (int, optional): if the number of samples is too much, the sampling process will terminate. Default to None.
            need_print (bool, optional): flag shows whether the total number of sampled pf will be printed. Defaults to False.

        Returns:
            ndarray: power flow settings.
        """        
        if generator_v_list is None: generator_v_list = np.arange(self.get_generator_number(sys_no))
        if generator_p_list is None: generator_p_list = self.get_generator_all_ctrl(None, sys_no)
        if load_p_list is None: load_p_list = np.arange(self.get_load_number(sys_no))
        if load_q_list is None: load_q_list = np.arange(self.get_load_number(sys_no))
        [lower_bounds, upper_bounds] = self.get_power_flow_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        gen_vmax = upper_bounds[:len(generator_v_list)]
        gen_vmin = lower_bounds[:len(generator_v_list)]
        gen_pmax = upper_bounds[len(generator_v_list):len(generator_v_list)+len(generator_p_list)]
        gen_pmin = lower_bounds[len(generator_v_list):len(generator_v_list)+len(generator_p_list)]
        slack_pmax = self.get_generator_all_pmax(self.__indexSlack)
        slack_pmin = self.get_generator_all_pmin(self.__indexSlack)
        load_pmax = upper_bounds[len(generator_v_list)+len(generator_p_list):len(generator_v_list)+len(generator_p_list)+len(load_p_list)]
        load_pmin = lower_bounds[len(generator_v_list)+len(generator_p_list):len(generator_v_list)+len(generator_p_list)+len(load_p_list)]
        load_qmax = upper_bounds[-len(load_q_list):]
        load_qmin = lower_bounds[-len(load_q_list):]
        sample_buffer = list()
        sample_total = 0
        gen_v = self.get_generator_all_v_set(generator_v_list, sys_no)
        gen_p = np.zeros(len(generator_p_list))
        sample_no = 0
        while sample_no < num:
            # load p max&min
            load_psum = -1.0
            while load_psum < (sum(gen_pmin) + sum(slack_pmin)) or load_psum > (sum(gen_pmax) + sum(slack_pmax)):
                load_p = load_pmin + (load_pmax - load_pmin) * self.__rng.random(len(load_pmax))
                load_psum = sum(load_p)
            # slack p
            load_psum = sum(load_p)
            s_pmin = max(load_psum - sum(gen_pmax), sum(slack_pmin))
            s_pmax = min(load_psum - sum(gen_pmin), sum(slack_pmax))
            s_pmin = s_pmin + (s_pmax - s_pmin) * self.__rng.random()
            s_pmax = s_pmin + (s_pmax - s_pmin) * self.__rng.random()
            # print('slack', s_pmin, s_pmax)
            # gen p
            for _ in range(5):
                gen_p.fill(0.)
                gen_order = self.__rng.choice(np.arange(len(gen_p)), len(generator_p_list), replace=False)
                load_psum = sum(load_p)
                # random slack
                for i in range(len(gen_p)):
                    gen_no = gen_order[i]
                    remain_gen = gen_order[i+1:] if i+1 < len(gen_p) else []
                    pmin = max(load_psum - sum(gen_pmax[remain_gen]) - s_pmax, gen_pmin[gen_no])
                    pmax = min(load_psum - sum(gen_pmin[remain_gen]) - s_pmin, gen_pmax[gen_no])
                    # print('generator', gen_no, [pmin, pmax])
                    gen_p[gen_no] = pmin + (pmax - pmin) * self.__rng.random()
                    load_psum -= gen_p[gen_no]
                # print('remaining', load_psum)
                # load q
                for _ in range(5):
                    load_q = load_qmin + (load_qmax - load_qmin) * self.__rng.random(len(load_qmax))
                    # gen v
                    for _ in range(5):
                        gen_v = gen_vmin + (gen_vmax - gen_vmin) * self.__rng.random(len(gen_vmax))
                        cur_status = np.concatenate([gen_v, gen_p, load_p, load_q])
                        self.set_power_flow_initiation(cur_status)
                        [converge, valid_slack, valid_v] = self.get_power_flow_status_check()
                        sample_total += 1
                        [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
                        if False not in [converge, valid_slack, valid_v]: break
                        if termination_num is not None and sample_total > termination_num: break
                    if False not in [converge, valid_slack, valid_v]: break
                    if termination_num is not None and sample_total > termination_num: break
                if False not in [converge, valid_slack, valid_v]: break
                if termination_num is not None and sample_total > termination_num: break
            if False not in [converge, valid_slack, valid_v]: 
                sample_no += 1
                sample_buffer.append(cur_status)
            if termination_num is not None and sample_total > termination_num: break
        if need_print: tqdm.write(f'total sample: {sample_total}, valid sample: {sample_no}')  
        return sample_buffer
    
    def get_power_flow_time_sequence(self, 
                                     num=1, 
                                     upper_range=5, 
                                     generator_v_list=None, 
                                     generator_p_list=None, 
                                     load_p_list=None, 
                                     load_q_list=None, 
                                     load_max=None, 
                                     load_min=None, 
                                     sys_no=-1, 
                                     check_converge=True, 
                                     check_slack=True, 
                                     check_voltage=True):
        """Power flow: get power flow time series samples.

        Args:
            num (int, optional): number of samples. Defaults to 1.
            upper_range (int, optional): Upper limit of the change rate between each time step. Defaults to 5.
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            load_max (float, optional): the load upper bound settings. Defaults to None. 
                None means loading the default settings shown by self.__load_pq_bounds.
                -1 means loading the current load settings.
                Others mean loading the original load settings and multiply the given load_max.
            load_min (float, optional): the load lower bound settings. Similar to load_max. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            check_converge (bool, optional): flag shows whether checking convergence. Defaults to True.
            check_slack (bool, optional): flag shows whether checking active power generation of slack generators. Defaults to True.
            check_voltage (bool, optional): flag shows whether checking nodal voltages. Defaults to True.

        Returns:
            ndarray: power flow settings.
        """        
        [lower_bounds, upper_bounds] = self.get_power_flow_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        vary_range = (upper_bounds - lower_bounds) * upper_range / 100
        bound_size = len(lower_bounds)
        sample_buffer = list()
        iter_buffer = list()
        r_vector = self.__rng.random(bound_size)
        [converge, valid_slack, valid_v] = [False, False, False]
        while (False in [converge, valid_slack, valid_v]):
            r_vector = self.__rng.random(bound_size)
            tmp_status = lower_bounds + (upper_bounds - lower_bounds) * r_vector
            self.set_power_flow_initiation(tmp_status, generator_v_list, generator_p_list, load_p_list, load_q_list)
            [converge, valid_slack, valid_v] = self.get_power_flow_status_check()
            [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
            iter_buffer.append(self.get_info_lf_iter())
            if False not in [converge, valid_slack, valid_v]:
                break
        pf_result = self.get_bus_all_lf_result()[:, 0]
        sample_buffer.append(pf_result)
        cur_status = tmp_status.copy()
        sample_total = 1
        for _ in range(num):
            [converge, valid_slack, valid_v] = [False, False, False]
            while (False in [converge, valid_slack, valid_v]):
                r_vector = self.__rng.random(bound_size) - 0.5
                tmp_status = cur_status + vary_range * r_vector
                self.set_power_flow_initiation(tmp_status, generator_v_list, generator_p_list, load_p_list, load_q_list)
                [converge, valid_slack, valid_v] = self.get_power_flow_status_check()
                [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
                iter_buffer.append(self.get_info_lf_iter())
                if False not in [converge, valid_slack, valid_v]:
                    break
            cur_status = tmp_status.copy()
            pf_result = self.get_bus_all_lf_result()[:, 0]
            sample_buffer.append(pf_result)
            sample_total += 1
            if sample_total >= num: break
        print(f'total sample: {sample_total}')  
        return sample_buffer

    # def get

    def get_pf_sample_all(self, num=1, generator_v_list=None, generator_p_list=None, load_p_list=None, load_q_list=None, load_max=None, load_min=None, sys_no=-1):
        """Power flow: power Flow Sampler, return initial state and convergence iteration number.

        Args:
            num (int, optional): Number of samples. Defaults to 1.
            generator_v_list (list, optional): list of generators with controllable v. Defaults to None.
            generator_p_list (list, optional): list of generators with controllable p. Defaults to None.
            load_p_list (list, optional): list of loads with controllable p. Defaults to None.
            load_q_list (list, optional): list of loads with controllable q. Defaults to None.
            load_max (float, optional): the load upper bound settings. Defaults to None. 
                None means loading the default settings shown by self.__load_pq_bounds.
                -1 means loading the current load settings.
                Others mean loading the original load settings and multiply the given load_max.
            load_min (float, optional): the load lower bound settings. Similar to load_max. Defaults to None.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list, list: sample list and iteration list (-1 for unconvergence)
        """        
        [lower_bounds, upper_bounds] = self.get_power_flow_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        bound_size = len(lower_bounds)
        sample_buffer = list()
        iter_buffer = list()
        for _ in range(num):
            r_vector = self.__rng.random(bound_size)
            cur_status = lower_bounds + (upper_bounds - lower_bounds) * r_vector
            # sample_buffer.append(cur_status)
            sample_buffer.append([cur_status, np.concatenate([cur_status[:10], r_vector[10:]])])
            self.set_power_flow_initiation(cur_status, generator_v_list, generator_p_list, load_p_list, load_q_list)
            iter_buffer.append(self.cal_power_flow_basic_nr())
        return sample_buffer, iter_buffer

    ################################################################################################
    # Transient Stability
    ################################################################################################
    def cal_transient_stability_simulation_ti_sv(self, start_time=0.0, contingency_no=0):
        """Transient stability simulation using the implicit trapezoidal method and sparse vector method. 

        Args:
            start_time (float, optional): The starting time of the simulation. Defaults to 0.0.
            contingency_no (int, optional): Contingency No. that determines which contingency needs simulation. See the config files for more information. Defaults to 0.

        Returns:
            int: The finish step of the simulation. Mostly, the maximum simulation step will be returned. 
            For example, the maximum simulation duration is 10 seconds and the integration step is 0.01, i.e., the maximum simulation step is 1000.
            Considering the starting step at instant t=0.00, the total number of simulation steps is 1000 + 1 = 1001.
        """        
        return self.__psDLL.cal_Transient_Stability_Simulation_TI_SV(start_time, contingency_no)
    
    # TODO api for other integration method and sparse vector method.

    def get_transient_stability_check_stability(self, maximum_delta=180.):
        """Transient stability: check stability of the simulation.

        Args:
            maximum_delta (float, optional): the acceptable maximum rotor angle. Defaults to 180..

        Returns:
            bool: flag shows whether the simulation is stable.
        """        
        std_result = self.get_acsystem_all_ts_result()[0]
        if std_result[:, 1].max() < maximum_delta:
            fin_step = self.get_info_ts_finish_step()
            if fin_step + 1 != self.get_info_ts_max_step():
                print(fin_step, std_result[:, 1].max())
                # raise Exception("stable and early finish, please check!")
            return True
        else:
            return False

    ################################################################################################
    # calculation information and calculation control
    ################################################################################################
    def get_info_lf_iter(self, sys_no=-1):
        """Get calculation information: number of iterations of power flow solution.

        Args:
            sys_no (int, optional): Asynchronous ac system No. Defaults to -1, which means the whole power system.

        Returns:
            int: The number of iterations of power flow solution.
        """        
        return self.__psDLL.get_Info_LF_Iter(sys_no)

    def get_info_ts_end_t(self):
        """Get calculation information: maximum simulation duration, usually set to 3 seconds for classic models or 10 seconds for detailed models.

        Returns:
            double/float64: TE, T end, maximum simulation duration.
        """        
        return self.__psDLL.get_Info_TE()

    def get_info_ts_delta_t(self):
        """Get calculation information: integration step of the simulaiton, usually set to 0.01 seconds, i.e., half cycle for electromechanical simulation. 

        Returns:
            double/float64: integration step.
        """        
        return self.__psDLL.get_Info_DT()

    def get_info_ts_max_step(self):
        """Get calculation information: the total number of simulation steps including the starting step.
           For example, the maximum simulation duration is 10 seconds and the integration step is 0.01, i.e., the maximum simulation step is 1000.
           Considering the starting step at instant t=0.00, the total number of simulation steps is 1000 + 1 = 1001.

        Returns:
            int: the total number of simulation steps.
        """        
        return self.__psDLL.get_Info_Max_Step()

    def get_info_ts_finish_step(self):
        """Get calculation information: finishing step of the transient simulation.
           Mostly, the total number of simulation steps will be returned. However, the simulation may end early for some reasons.
           The simulation may end because of convergence problems, e.g., the simulation does not converge at step 432, then 432 will be returned.
           TODO A setting flag will be set for whether initiating the early termination of simulation, which means the simulation will stop whenever the system lose stability. 

        Returns:
            int: The finishing step of the simulation. 
        """        
        self.__cur_total_step = self.__psDLL.get_Info_Finish_Step()
        return self.__cur_total_step

    def get_info_fault_step_sequence(self, contingency_no=0):
        """Get calculation information: the fault step sequence, an array of fault steps will be returned to show the instant of faults and disturbances.
           For example, the starting time is 0.00, the integration step is 0.01, and a three-phase short-circuit fault happens at instant 1.0 and clear at instant 1.1.
           The returned step array will be ndarray([0, 100, 110], dtype=int32)

        Args:
            contingency_no (int, optional): Contingency No. that determines which contingency needs simulation. See the config files for more information. Defaults to 0.

        Returns:
            int numpy array: fault step sequence.
        """        
        assert self.__psDLL.get_Info_Fault_Step_Sequence(self.__doubleBuffer, contingency_no) is True, "get fault step sequence failed, please check!"
        # [0] is n_fault_step, [1:] is step sequences
        n_fault_step = int(self.__doubleBuffer[0])
        return self.__doubleBuffer[1:n_fault_step].astype(np.int32)

    def get_info_fault_time_sequence(self, contingency_no=0):
        """Get calculation information: the fault time sequence, an array of instants of faults and disturbances will be returned.
           For example, the starting time is 0.00 and a three-phase short-circuit fault happens at instant 1.0 and clear at instant 1.1.
           The returned step array will be ndarray([0.0, 1.0, 1.1], dtype=float32)

        Args:
            contingency_no (int, optional): Contingency No. that determines which contingency needs simulation. See the config files for more information. Defaults to 0.

        Returns:
            float32 numpy array: fault time sequence.
        """        
        assert self.__psDLL.get_Info_Fault_Time_Sequence(self.__doubleBuffer, contingency_no) is True, "get fault time sequence failed, please check!"
        # [0] is n_fault_step, [1:] is time sequences
        n_fault_step = int(self.__doubleBuffer[0])
        return self.__doubleBuffer[1:n_fault_step].astype(np.float32)

    def set_info_ts_step_network_state(self, step: int, is_real_step=True, sys_no=-1):
        """Set calculation information: set the network state to a certain integration step. 
           If is_real_step is true, the step is the real integration step, e.g., when integration step is 0.01s, t=0.0s is step 0, t=0.15 is step 15, t=2.34s is step 234, etc.
           If is_real_step is false, the step is the actual step in the simulator PSOPS. 
           For example, when integration step is 0.01s and a three-phase fault happens at t=0.2s and clear at t=0.3s, t=0.0s is step 0, t=0.21 is step 22, t=0.31s is step 33, etc.
           The setting range is defined by the sys_no, sys_no=-1 means the whole network. 

        Args:
            step (int): The target step.
            is_real_step (bool, optional): A flag showing whether the step is a real one or the actual step in simulation. Defaults to True.
            sys_no (int, optional): System range. Defaults to -1.
        """
        assert self.__psDLL.set_Info_TS_Step_Network_State(step, is_real_step, sys_no) is True, "Step No or AC System No error. Please check!"

    def set_info_ts_step_element_state(self, step: int, is_real_step=True, sys_no=-1):
        """Set calculation information: set the component state to a certain integration step. 
           If is_real_step is true, the step is the real integration step, e.g., when integration step is 0.01s, t=0.0s is step 0, t=0.15 is step 15, t=2.34s is step 234, etc.
           If is_real_step is false, the step is the actual step in the simulator PSOPS. 
           For example, when integration step is 0.01s and a three-phase fault happens at t=0.2s and clear at t=0.3s, t=0.0s is step 0, t=0.21 is step 22, t=0.31s is step 33, etc.
           The setting range is defined by the sys_no, sys_no=-1 means the whole network. 

        Args:
            step (int): The target step.
            is_real_step (bool, optional): A flag showing whether the step is a real one or the actual step in simulation. Defaults to True.
            sys_no (int, optional): System range. Defaults to -1.
        """        
        assert self.__psDLL.set_Info_TS_Step_Element_State(step, is_real_step, sys_no) is True, "Step No or AC System No error. Please check!"

    def set_info_ts_step_all_state(self, step: int, is_real_step=True, sys_no=-1):
        """Set calculation information: set the all the state including components' states and network state to a certain integration step. 
           If is_real_step is true, the step is the real integration step, e.g., when integration step is 0.01s, t=0.0s is step 0, t=0.15 is step 15, t=2.34s is step 234, etc.
           If is_real_step is false, the step is the actual step in the simulator PSOPS. 
           For example, when integration step is 0.01s and a three-phase fault happens at t=0.2s and clear at t=0.3s, t=0.0s is step 0, t=0.21 is step 22, t=0.31s is step 33, etc.
           The setting range is defined by the sys_no, sys_no=-1 means the whole network. 

        Args:
            step (int): The target step.
            is_real_step (bool, optional): A flag showing whether the step is a real one or the actual step in simulation. Defaults to True.
            sys_no (int, optional): System range. Defaults to -1.
        """        
        assert self.__psDLL.set_Info_TS_Step_All_State(step, is_real_step, sys_no) is True, "Step No or AC System No error. Please check!"

    ################################################################################################
    # ac systems
    ################################################################################################
    def get_acsystem_number(self):
        """Get system information: the number of asynchronous ac systems.

        Returns:
            int: the number of asynchronous ac systems.
        """
        return self.__nACSystem

    def get_acsystem_ts_cur_step_result(self, sys_no=-1, buffer=None, rt=True):
        """Get system information: current step result of system level variables including:
        [0] time step
        [1] maximum rotor angle difference
        [2] minimum frequency/rotation speed
        [3] maximum frequency/rotation speed
        [4] minimum nodal voltage
        [5] maximum nodal voltage
        The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_acsystem_ts_step_result().

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: the simulation result of current step of system variables of a asynchonous ac system determined by the sys_no. Array shape is (n_system, 6).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_ACSystem_TS_CurStep_Result(buffer, sys_no) == True, "get ts system variable result failed, please check!"
        if rt is True:
            n_system = self.__nACSystem if sys_no == -1 else 1
            return buffer[:n_system*6].reshape(n_system, 6).astype(np.float32)

    def get_acsystem_ts_step_result(self, step: int, sys_no=-1):
        """Get system information: result of system level variables at step. 
           Also see self.get_acsystem_ts_cur_step_result().

        Args:
            step (int): real integration step.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: the simulation step of current step of system variables of asynchonous ac system sys_no. Array shape is (n_system, 6).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_acsystem_ts_cur_step_result(sys_no)

    # transient stability, all steps, all system variable, result, time, max delta, min freq, max freq, min vol, max vol
    def get_acsystem_all_ts_result(self, sys_no=-1):
        """Get system information: all steps' results of system level variables. 
           Also see self.get_acsystem_ts_cur_step_result().

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: the simulation results of all steps of system variables of a asynchonous ac system sys_no. Array shape is (n_system, total_step, 6).
        """        
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        if total_step > self.get_info_ts_max_step(): print(f'max step exceed: {total_step}!!!!!!!!!!!!!!!!!!!!')
        assert self.__psDLL.get_ACSystem_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all system variable results failed!"
        n_system = self.__nACSystem if sys_no == -1 else 1
        return self.__doubleBuffer[0:n_system * total_step * 6].reshape(n_system, total_step, 6).astype(np.float32)

    ################################################################################################
    # buses
    ################################################################################################
    def get_bus_number(self, sys_no=-1):
        """Get bus information: the number of buses in asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the number of buses.
        """        
        n_bus = self.__nBus if sys_no == -1 else self.__psDLL.get_Bus_Number(sys_no)
        assert n_bus >= 0, 'total number of bus wrong, please check!'
        return n_bus

    def get_bus_name(self, bus_no: int, sys_no=-1):
        """Get bus information: the bus name of bus bus_no of asynchronous system sys_no.

        Args:
            bus_no (int): the bus No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            gbk str: the bus name string.
        """        
        if sys_no == -1:
            return self.__allBusName[bus_no]
        bus_name = self.__psDLL.get_Bus_Name(bus_no, sys_no)
        assert bus_name is not None,  "bus name is empty, please check sys/bus no!"
        return string_at(bus_name, -1).decode('gbk')

    def get_bus_all_name(self, bus_list=None, sys_no=-1):
        """Get bus information: all the bus names of buses in the bus_list of asynchronous system sys_no.

        Args:
            bus_list (list int, optional): the list of buses whose names are needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray str: the array of bus names.
        """        
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int32) if bus_list is None else bus_list
        if sys_no == -1: return self.__allBusName[bus_list]
        bus_names = list()
        for bus_no in bus_list: bus_names.append(self.get_bus_name(bus_no, sys_no))
        return np.array(bus_names)

    def get_bus_no(self, name: str):
        """Get bus information: the bus No. according to bus name.

        Args:
            name (str): bus name.

        Returns:
            int: bus No. The node ordering algorithm will influence the bus No. TODO change node ordering algorithm through python API.
        """        
        bus_no = np.where(self.__allBusName == name)[0]
        assert len(bus_no) == 1, "bus name duplication"
        return bus_no[0]

    def get_bus_sys_no(self, bus_no: int):
        """Get bus information: the asynchronous system No. of the bus bus_no.

        Args:
            bus_no (int): the bus No. in asynchronous system sys_no.

        Returns:
            int: asynchronous system No.
        """        
        sys_no = self.__psDLL.get_Bus_Sys_No(bus_no)
        assert sys_no >= 0, "bus asynchronous system detection failed!"
        return sys_no

    def get_bus_all_sys_no(self, bus_list=None):
        """Get bus information: all the asynchronous system NO. of buses in the bus_list.

        Args:
            bus_list (list int, optional): the list of buses whose asynchronous system No. is needed. Defaults to None, which means all the buses.

        Returns:
            ndarray int: an array of the asynchronous system No.
        """        
        bus_list = np.arange(self.__nBus, dtype=np.int32) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__intBuffer[index] = self.get_bus_sys_no(bus_no)
        return self.__intBuffer[:len(bus_list)].astype(np.int32)

    def get_bus_vmax(self, bus_no: int, sys_no=-1):
        """Get bus information: the upper bound of bus voltage.

        Args:
            bus_no (int): the bus No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the upper bound of bus voltage.
        """        
        vmax = self.__psDLL.get_Bus_VMax(bus_no, sys_no)
        assert vmax > -1.0e10, "vmax wrong, please check!"
        return vmax

    def get_bus_all_vmax(self, bus_list=None, sys_no=-1):
        """Get bus Information: all the upper bounds of buses in the bus_list of asynchronous system sys_no.

        Args:
            bus_list (list int, optional): the list of buses whose upper bound of bus voltage is needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of the upper bounds of buses.
        """        
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int32) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__doubleBuffer[index] = self.get_bus_vmax(bus_no, sys_no)
        return self.__doubleBuffer[:len(bus_list)].astype(np.float32)

    def get_bus_vmin(self, bus_no: int, sys_no=-1):
        """Get bus information: the lower bound of bus voltage.

        Args:
            bus_no (int): the bus No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the lower bound of bus voltage.
        """        
        vmin = self.__psDLL.get_Bus_VMin(bus_no, sys_no)
        assert vmin > -1.0e10, "vmin wrong, please check!"
        return vmin

    def get_bus_all_vmin(self, bus_list=None, sys_no=-1):
        """Get bus Information: all the lower bounds of buses in the bus_list of asynchronous system sys_no.

        Args:
            bus_list (list int, optional): the list of buses whose upper bound of bus voltage is needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of the lower bounds of buses.
        """        
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int32) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__doubleBuffer[index] = self.get_bus_vmin(bus_no, sys_no)
        return self.__doubleBuffer[:len(bus_list)].astype(np.float32)

    def get_bus_lf_result(self, bus_no: int, sys_no=-1, buffer=None, rt=True):
        """Get bus information: the power flow result of bus bus_no of asychronous system sys_no.
           [0] bus voltage amplitude
           [1] bus voltage phase
           [2] active power generation
           [3] reactive power generation
           [4] active power load
           [5] reactive power load

        Args:
            bus_no (int): the bus No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of bus power flow result. Array shape is (6,).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Bus_LF_Result(buffer, bus_no, sys_no) == True, "get lf bus result failed, please check!"
        if rt is True:
            return buffer[:6].astype(np.float32)

    # load flow all bus result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_all_lf_result(self, bus_list=None, sys_no=-1):
        """Get bus information: all the power flow results of buses in the bus_list of asychronous system sys_no.
           Also see self.get_bus_lf_result().

        Args:
            bus_list (list int, optional): the list of buses whose upper bound of bus voltage is needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of all the power flow results of buses. Array shape is (len(bus_list), 6). 
        """        
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int32) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.get_bus_lf_result(bus_no, sys_no, self.__doubleBuffer[index*6:], False)
        return self.__doubleBuffer[:len(bus_list)*6].reshape(len(bus_list), 6).astype(np.float32)

    def get_bus_ts_cur_step_result(self, bus_no: int, sys_no=-1, buffer=None, rt=True):
        """Get bus information: simulation result of bus bus_no of asynchronous system sys_no at current step. 
           [0] bus voltage amplitude
           [1] bus voltage phase
           [2] active power generation
           [3] reactive power generation
           [4] active power load
           [5] reactive power load
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_bus_ts_step_result().

        Args:
            bus_no (int): bus No. 
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of bus simulation result. Array shape is (6,).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Bus_TS_CurStep_Result(buffer, bus_no, sys_no) == True, "get ts bus result failed, please check!"
        if rt is True:
            return buffer[:6].astype(np.float32)

    def get_bus_all_ts_cur_step_result(self, bus_list=None, sys_no=-1):
        """Get bus information: all the simulation results of buses in the bus_list of asychronous system sys_no at current step.
           Also see self.get_bus_ts_cur_step_result().

        Args:
            bus_list (list int, optional): the list of buses whose upper bound of bus voltage is needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of buses simulation results. Array shape is (len(bus_list), 6).
        """        
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int32) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.get_bus_ts_cur_step_result(bus_no, sys_no, self.__doubleBuffer[index*6:], False)
        return self.__doubleBuffer[:len(bus_list)*6].reshape(len(bus_list), 6).astype(np.float32)

    def get_bus_ts_step_result(self, step: int, bus_no: int, sys_no=-1):
        """Get bus information: simulation result of bus bus_no of asynchronous system sys_no at step. 
           Also see self.get_bus_ts_cur_step_result().

        Args:
            step (int): real integration step.
            bus_no (int): bus No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of bus simulation result. Array shape is (6,).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_bus_ts_cur_step_result(bus_no, sys_no)

    def get_bus_all_ts_step_result(self, step: int, bus_list=None, sys_no=-1):
        """Get bus information: all the simulation results of buses in the bus_list of asychronous system sys_no at step.
        Also see self.get_bus_all_ts_cur_step_result().

        Args:
            step (int): real integration step.
            bus_list (list int, optional): the list of buses whose upper bound of bus voltage is needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of buses simulation results. Array shape is (len(bus_list), 6).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_bus_all_ts_cur_step_result(bus_list, sys_no)

    def get_bus_all_ts_result(self, bus_list=None, sys_no=-1):
        """Get bus information: all the simulation results of buses in the bus_list of asychronous system sys_no.

        Args:
            bus_list (list int, optional): the list of buses whose upper bound of bus voltage is needed. Defaults to None, which means all the buses.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of buses simulation results. Array shape is (total_step, len(bus_list), 6).
        """      
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int32) if bus_list is None else bus_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Bus_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all bus results failed!"
        all_result = self.__doubleBuffer[0:total_step * 6 * self.get_bus_number(sys_no)].reshape(total_step, self.get_bus_number(sys_no), 6)
        return all_result[:, bus_list, :].astype(np.float32)

    ################################################################################################
    # aclines
    ################################################################################################
    def get_acline_number(self, sys_no=-1):
        """Get acline information: the number of aclines of asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the number of aclines.
        """        
        n_acline = self.__nACLine if sys_no == -1 else self.__psDLL.get_ACLine_Number(sys_no)
        assert n_acline >= 0, 'total number of acline wrong, please check!'
        return n_acline

    def get_acline_i_no(self, acline_no: int, sys_no=-1):
        """Get acline information: the bus No. of bus i of acline acline_no of synchronous system sys_no. 
        An acline has two terminals, i.e., bus i at the starting terminal and bus j at the ending termial.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: bus No. of bus i.
        """        
        i_no = self.__psDLL.get_ACLine_I_No(acline_no, sys_no)
        assert i_no >= 0, "acline i no wrong, please check!"
        return i_no

    def get_acline_all_i_no(self, acline_list=None, sys_no=-1):
        """Get acline information: all the bus No. of bus i of aclines in acline_list of asynchronous system sys_no.
        Also see self.get_acline_i_no().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int32: an array of bus No. of bus i.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_i_no(acline_no, sys_no)
        return self.__intBuffer[:len(acline_list)].astype(np.int32)

    def get_acline_j_no(self, acline_no: int, sys_no=-1):
        """Get acline information: the bus No. of bus j of acline acline_no of synchronous system sys_no.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: bus No. of bus j.
        """        
        j_no = self.__psDLL.get_ACLine_J_No(acline_no, sys_no)
        assert j_no >= 0, "acline j no, please check!"
        return j_no

    def get_acline_all_j_no(self, acline_list=None, sys_no=-1):
        """Get acline information: all the bus No. of bus j of aclines in acline_list of asynchronous system sys_no.
           Also see self.get_acline_j_no().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int32: an array of bus No. of bus j.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_j_no(acline_no, sys_no)
        return self.__intBuffer[:len(acline_list)].astype(np.int32)

    def get_acline_i_name(self, acline_no: int, sys_no=-1):
        """Get acline information: the bus name of bus i of acline acline_no of asynchronous system sys_no.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: bus name of bus i.
        """        
        return self.get_bus_name(self.get_acline_i_no(acline_no, sys_no), sys_no)

    def get_acline_all_i_name(self, acline_list=None, sys_no=-1):
        """Get acline information: all the bus names of bus i of aclines in acline_list of asynchronous system sys_no.
           Also see self.get_acline_i_name().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray, str: an array of bus names of bus i.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        return self.get_bus_all_name(None, sys_no)[self.get_acline_all_i_no(acline_list)]

    def get_acline_j_name(self, acline_no: int, sys_no=-1):
        """Get acline information: the bus name of bus j of acline acline_no of asynchronous system sys_no.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: bus name of bus j.
        """        
        return self.get_bus_name(self.get_acline_j_no(acline_no, sys_no), sys_no)

    def get_acline_all_j_name(self, acline_list=None, sys_no=-1):
        """Get acline information: all the bus names of bus j of aclines in acline_list of asynchronous system sys_no.
           Also see self.get_acline_j_name().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray, str: an array of bus names of bus j.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        return self.get_bus_all_name(None, sys_no)[self.get_acline_all_j_no(acline_list)]

    def get_acline_sys_no(self, acline_no: int):
        """Get acline information: the asynchronous system No. of acline acline_no.

        Args:
            acline_no (int): the acline No. in all the aclines in the power system.

        Returns:
            int: the asynchronous system No.
        """        
        sys_no = self.__psDLL.get_ACLine_Sys_No(acline_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    def get_acline_all_sys_no(self, acline_list=None):
        """Get acline information: all the asynchronous system No. of aclines in acline_list.
           Also see self.get_acline_sys_no().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.

        Returns:
            ndarray int: an array of asynchronous system No. 
        """        
        acline_list = np.arange(self.get_acline_number(), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_sys_no(acline_no)
        return self.__intBuffer[:len(acline_list)].astype(np.int32)

    def get_acline_NO(self, acline_no: int, sys_no=-1):
        """Get acline information: the internal No. of acline acline_no of asynchronous system sys_no.
           This value is determined in the original computation datafile.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the internal No. of acline.
        """        
        a_NO = self.__psDLL.get_ACLine_No(acline_no, sys_no)
        assert a_NO >= 0, "acline No is wrong, please check!"
        return a_NO

    def get_acline_all_NO(self, acline_list=None, sys_no=-1):
        """Get acline information: all the internal No. of aclines in acline_list of asynchronous system sys_no.
           Also see self.get_acline_NO().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray: an array of internal No. of aclines.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_NO(acline_no, sys_no)
        return self.__intBuffer[:len(acline_list)]

    def get_acline_current_capacity(self, acline_no: int, sys_no=-1):
        """Get acline information: the line current capacity of acline acline_no of asynchronous system sys_no.
           This value is determined in the original computation datafile.
           TODO change this value by python API.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the internal No. of acline.
        """        
        current_capacity = self.__psDLL.get_ACLine_Current_Capacity(acline_no, sys_no)
        assert current_capacity >= 0.0, "acline current capacity wrong, please check!"
        return current_capacity

    def get_acline_all_current_capacity(self, acline_list=None, sys_no=-1):
        """Get acline information: all the current capacities of aclines in acline_list of asynchronous system sys_no.
           Also see self.get_acline_current_capacity().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray, float64: an array of current capacities of aclines.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__doubleBuffer[index] = self.get_acline_current_capacity(acline_no, sys_no)
        return self.__doubleBuffer[:len(acline_list)].astype(np.float32)

    def get_acline_lf_result(self, acline_no: int, sys_no=-1, buffer=None, rt=True):
        """Get acline information: the power flow result of acline acline_no of asychronous system sys_no.
           [0] active power injected to bus i
           [1] reactive power injected to bus i
           [2] active power load injected to bus j
           [3] reactive power load injected to bus j

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of acline power flow result. Array shape is (4,).
        """    
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_ACLine_LF_Result(buffer, acline_no, sys_no) == True, "get lf acline result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    def get_acline_all_lf_result(self, acline_list=None, sys_no=-1):
        """Get acline information: all the power flow results of aclines in the bus_list of asychronous system sys_no.
           Also see self.get_acline_lf_result().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of all acline power flow results. Array shape is (len(acline_list), 4).
        """
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.get_acline_lf_result(acline_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(acline_list)*4].reshape(len(acline_list), 4).astype(np.float32)

    def get_acline_ts_cur_step_result(self, acline_no: int, sys_no=-1, buffer=None, rt=True):
        """Get acline information: simulation result of acline acline_no of asychronous system sys_no at current step.
           [0] active power injected to bus i
           [1] reactive power injected to bus i
           [2] active power load injected to bus j
           [3] reactive power load injected to bus j
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_acline_ts_step_result().

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of bus simulation result. Array shape is (4,).
        """    
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_ACLine_TS_CurStep_Result(buffer, acline_no, sys_no) == True, "get ts acline result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    def get_acline_all_ts_cur_step_result(self, acline_list=None, sys_no=-1):
        """Get acline information: all the simulation results of aclines in acline_list of asychronous system sys_no at current step.
           Also see self.get_acline_ts_cur_step_result().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of aclines simulation results. Array shape is (len(acline_list), 4).
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.get_acline_ts_cur_step_result(acline_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(acline_list)*4].reshape(len(acline_list), 4).astype(np.float32)

    def get_acline_ts_step_result(self, step: int, acline_no: int, sys_no=-1):
        """Get acline information: simulation result of acline acline_no of asychronous system sys_no at step.
           Also see self.get_acline_ts_cur_step_result().

        Args:
            step (int): real integration step.
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of bus simulation result. Array shape is (4,).
        """          
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_acline_ts_cur_step_result(acline_no, sys_no)

    def get_acline_all_ts_step_result(self, step: int, acline_list=None, sys_no=-1):
        """Get acline information: all the simulation results of aclines in acline_list of asychronous system sys_no at step.
           Also see self.get_acline_all_ts_cur_step_result().

        Args:
            step (int): real integration step.
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of aclines simulation results. Array shape is (len(acline_list), 4).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_acline_all_ts_cur_step_result(acline_list, sys_no)

    def get_acline_all_ts_result(self, acline_list=None, sys_no=-1):
        """Get acline information: all the simulation results of aclines in the acline_list of asychronous system sys_no.
           Also see self.get_acline_all_ts_cur_step_result().

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of aclines simulation results. Array shape is (total_step, len(acline_list), 4).
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_ACLine_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all acline results failed!"
        all_result = self.__doubleBuffer[0:total_step * 4 * self.get_acline_number(sys_no)].reshape(total_step, self.get_acline_number(sys_no), 4)
        return all_result[:, acline_list, :].astype(np.float32)

    def get_acline_info(self, acline_no: int, sys_no=-1):
        """Get acline information: acline name information of acline acline_no of asychronous system sys_no.

        Args:
            acline_no (int): acline No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list: a list of acline information, [bus i name, bus j name, internal No.].
        """        
        return [self.get_acline_i_name(acline_no, sys_no), self.get_acline_j_name(acline_no, sys_no), self.get_acline_NO(acline_no, sys_no)]
    
    def get_acline_all_info(self, acline_list=None, sys_no=-1):
        """Get acline information: all acline name information of aclines in acline_list of asychronous system sys_no.

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list: a list of aclines information, [[bus i names], [bus j names], [internal No.]]
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        return [self.get_acline_all_i_name(acline_list, sys_no), self.get_acline_all_j_name(acline_list, sys_no), self.get_acline_all_NO(acline_list, sys_no)]

    ################################################################################################
    # transformers
    ################################################################################################
    def get_transformer_number(self, sys_no=-1):
        """Get transformer information: the number of transformer in asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the number of transformers.
        """        
        if sys_no == -1:
            return self.__nTransformer
        n_transformer = self.__psDLL.get_Transformer_Number(sys_no)
        assert n_transformer >= 0, 'total number of transformer wrong, please check!'
        return n_transformer

    def get_transformer_i_no(self, transformer_no: int, sys_no=-1):
        """Get transformer information: the bus No. of bus i of transformer transformer_no of synchronous system sys_no. 
        An transformer has two terminals, i.e., bus i at the starting terminal and bus j at the ending termial.

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: bus No. of bus i.
        """        
        i_no = self.__psDLL.get_Transformer_I_No(transformer_no, sys_no)
        assert i_no >= 0, "transformer i no wrong, please check!"
        return i_no

    def get_transformer_all_i_no(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the bus No. of bus i of transformers in transformer_list of asynchronous system sys_no.
        Also see self.get_transformer_i_no().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int32: an array of bus No. of bus i.
        """                
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_i_no(transformer_no, sys_no)
        return self.__intBuffer[:len(transformer_list)].astype(np.int32)

    def get_transformer_j_no(self, transformer_no: int, sys_no=-1):
        """Get transformer information: the bus No. of bus j of transformer transformer_no of synchronous system sys_no. 

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: bus No. of bus j.
        """        
        j_no = self.__psDLL.get_Transformer_J_No(transformer_no, sys_no)
        assert j_no >= 0, "transformer j no, please check!"
        return j_no

    def get_transformer_all_j_no(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the bus No. of bus j of transformers in transformer_list of asynchronous system sys_no.
        Also see self.get_transformer_j_no().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int32: an array of bus No. of bus j.
        """                
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_j_no(transformer_no, sys_no)
        return self.__intBuffer[:len(transformer_list)].astype(np.int32)

    def get_transformer_i_name(self, transformer_no: int, sys_no=-1):
        """Get transformer information: the bus name of bus i of transformer transformer_no of asynchronous system sys_no.

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: bus name of bus i.
        """        
        return self.get_bus_name(self.get_transformer_i_no(transformer_no, sys_no), sys_no)

    def get_transformer_all_i_name(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the bus names of bus i of transformers in transformer_list of asynchronous system sys_no.
           Also see self.get_transformer_i_name().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray, str: an array of bus names of bus i.
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        return self.get_bus_all_name(None, sys_no)[self.get_transformer_all_i_no(transformer_list, sys_no)]

    def get_transformer_j_name(self, transformer_no: int, sys_no=-1):
        """Get transformer information: the bus name of bus j of transformer transformer_no of asynchronous system sys_no.

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: bus name of bus j.
        """        
        return self.get_bus_name(self.get_transformer_j_no(transformer_no, sys_no), sys_no)

    def get_transformer_all_j_name(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the bus names of bus j of transformers in transformer_list of asynchronous system sys_no.
           Also see self.get_transformer_j_name().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray, str: an array of bus names of bus j.
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        return self.get_bus_all_name(None, sys_no)[self.get_transformer_all_j_no(transformer_list, sys_no)]

    def get_transformer_sys_no(self, transformer_no: int):
        """Get transformer information: the asynchronous system No. of transformer transformer_no.

        Args:
            transformer_no (int): the transformer No. in all the transformers in the power system.

        Returns:
            int: the asynchronous system No.
        """        
        sys_no = self.__psDLL.get_Transformer_Sys_No(transformer_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    def get_transformer_all_sys_no(self, transformer_list=None):
        """Get transformer information: all the asynchronous system No. of transformers in transformer_list.
           Also see self.get_transformer_sys_no().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.

        Returns:
            ndarray int: an array of asynchronous system No. 
        """        
        transformer_list = np.arange(self.get_transformer_number(), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_sys_no(transformer_no)
        return self.__intBuffer[:len(transformer_list)].astype(np.int32)

    def get_transformer_NO(self, transformer_no: int, sys_no=-1):
        """Get transformer information: the internal No. of transformer transformer_no of asynchronous system sys_no.
           This value is determined in the original computation datafile.

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the internal No. of transformer.
        """        
        a_NO = self.__psDLL.get_Transformer_No(transformer_no, sys_no)
        assert a_NO >= 0, "transformer No is wrong, please check!"
        return a_NO

    def get_transformer_all_NO(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the internal No. of transformers in transformer_list of asynchronous system sys_no.
           Also see self.get_transformer_NO().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray: an array of internal No. of transformers.
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_NO(transformer_no, sys_no)
        return self.__intBuffer[:len(transformer_list)]

    def get_transformer_current_capacity(self, transformer_no: int, sys_no=-1):
        """Get transformer information: the line current capacity of transformer transformer_no of asynchronous system sys_no.
           This value is determined in the original computation datafile.
           TODO change this value by python API.

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the internal No. of transformer. 
        """        
        current_capacity = self.__psDLL.get_Transformer_Current_Capacity(transformer_no, sys_no)
        assert current_capacity > 0.0, "transformer current capacity wrong, please check!"
        return current_capacity

    def get_transformer_all_current_capacity(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the current capacities of transformers in transformer_list of asynchronous system sys_no.
           Also see self.get_transformer_current_capacity().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray, float64: an array of current capacities of transformers.
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__doubleBuffer[index] = self.get_transformer_current_capacity(transformer_no, sys_no)
        return self.__doubleBuffer[:len(transformer_list)].astype(np.float32)

    def get_transformer_lf_result(self, transformer_no: int, sys_no=-1, buffer=None, rt=True):
        """Get transformer information: the power flow result of transformer transformer_no of asychronous system sys_no.
           [0] active power injected to bus i
           [1] reactive power injected to bus i
           [2] active power load injected to bus j
           [3] reactive power load injected to bus j

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of transformer power flow result. Array shape is (4,).
        """    
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Transformer_LF_Result(buffer, transformer_no, sys_no) == True, "get lf transformer result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    def get_transformer_all_lf_result(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the power flow results of transformers in the bus_list of asychronous system sys_no.
           Also see self.get_transformer_lf_result().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of all transformer power flow results. Array shape is (len(transformer_list), 4).
        """
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.get_transformer_lf_result(transformer_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(transformer_list)*4].reshape(len(transformer_list), 4).astype(np.float32)

    def get_transformer_ts_cur_step_result(self, transformer_no: int, sys_no=-1, buffer=None, rt=True):
        """Get transformer information: simulation result of transformer transformer_no of asychronous system sys_no at current step.
           [0] active power injected to bus i
           [1] reactive power injected to bus i
           [2] active power load injected to bus j
           [3] reactive power load injected to bus j
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_transformer_ts_step_result().

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of bus simulation result. Array shape is (4,).
        """    
        buffer = self.__doubleBuffer if buffer is None else buffer 
        assert self.__psDLL.get_Transformer_TS_CurStep_Result(buffer, transformer_no, sys_no) == True, "get ts transformer result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    def get_transformer_all_ts_cur_step_result(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the simulation results of transformers in transformer_list of asychronous system sys_no at current step.
           Also see self.get_transformer_ts_cur_step_result().

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of transformers simulation results. Array shape is (len(transformer_list), 4).
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.get_transformer_ts_cur_step_result(transformer_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(transformer_list)*4].reshape(len(transformer_list), 4).astype(np.float32)

    def get_transformer_ts_step_result(self, step: int, transformer_no: int, sys_no=-1):
        """Get transformer information: simulation result of transformer transformer_no of asychronous system sys_no at step.
           Also see self.get_transformer_ts_cur_step_result().

        Args:
            step (int): real integration step.
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of bus simulation result. Array shape is (4,). 
        """          
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_transformer_ts_cur_step_result(transformer_no, sys_no)

    def get_transformer_all_ts_step_result(self, step: int, transformer_list=None, sys_no=-1):
        """Get transformer information: all the simulation results of transformers in transformer_list of asychronous system sys_no at step.
           Also see self.get_transformer_all_ts_cur_step_result().

        Args:
            step (int): real integration step.
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of transformers simulation results. Array shape is (len(transformer_list), 4).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_transformer_all_ts_cur_step_result(transformer_list, sys_no)

    def get_transformer_all_ts_result(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all the simulation results of transformers in the transformer_list of asychronous system sys_no.

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of transformers simulation results. Array shape is (total_step, len(transformer_list), 4).
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Transformer_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all transformer results failed!"
        all_result = self.__doubleBuffer[0:total_step * 4 * self.get_transformer_number(sys_no)].reshape(total_step, self.get_transformer_number(sys_no), 4)
        return all_result[:, transformer_list, :].astype(np.float32)

    def get_transformer_info(self, transformer_no: int, sys_no=-1):
        """Get transformer information: transformer name information of transformer transformer_no of asychronous system sys_no.

        Args:
            transformer_no (int): transformer No. in asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list: a list of transformer information, [bus i name, bus j name, internal No.]. 
        """        
        return [self.get_transformer_i_name(transformer_no, sys_no), self.get_transformer_j_name(transformer_no, sys_no), self.get_transformer_NO(transformer_no, sys_no)]
    
    def get_transformer_all_info(self, transformer_list=None, sys_no=-1):
        """Get transformer information: all transformer name information of transformers in transformer_list of asychronous system sys_no.

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list: a list of transformers information, [[bus i names], [bus j names], [internal No.]]
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        return [self.get_transformer_all_i_name(transformer_list, sys_no), self.get_transformer_all_j_name(transformer_list, sys_no), self.get_transformer_all_NO(transformer_list, sys_no)]

    ################################################################################################
    # generators
    ################################################################################################
    def get_generator_number(self, sys_no=-1):
        """Get generator information: the number of generators in asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the number of generators.
        """        
        if sys_no == -1:
            return self.__nGenerator
        n_generator = self.__psDLL.get_Generator_Number(sys_no)
        assert n_generator >= 0, 'total number of generator wrong, please check!'
        return n_generator

    def get_generator_bus_no(self, generator_no: int, sys_no=-1):
        """Get generator information: the bus No. of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: bus No. of the generator.
        """        
        bus_no = self.__psDLL.get_Generator_Bus_No(generator_no, sys_no)
        assert bus_no >= 0, "generator i no wrong, please check!"
        return bus_no

    def get_generator_all_bus_no(self, generator_list=None, sys_no=-1):
        """Get generator information: all the bus No. of generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.

        Returns:
            ndarray int: an array of the bus No. of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__intBuffer[index] = self.get_generator_bus_no(generator_no, sys_no)
        return self.__intBuffer[:len(generator_list)].astype(np.int32)

    def get_generator_bus_name(self, generator_no: int, sys_no=-1):
        """Get generator information: the bus name of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: bus name of the generator.
        """        
        return self.get_bus_name(self.get_generator_bus_no(generator_no, sys_no), sys_no)

    def get_generator_all_bus_name(self, generator_list=None, sys_no=-1):
        """Get generator information: all the bus names of generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.

        Returns:
            ndarray str: an array of the bus names of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        return self.get_bus_all_name(None, sys_no)[self.get_generator_all_bus_no(generator_list, sys_no)]

    def get_generator_sys_no(self, generator_no: int):
        """Get generator information: the asynchronous system No. of generator generator_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.

        Returns:
            int: asynchronous system No. of the generator.
        """        
        sys_no = self.__psDLL.get_Generator_Sys_No(generator_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    def get_generator_all_sys_no(self, generator_list=None):
        """Get generator information: all the asynchronous system No. of generators in the generator_list.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.

        Returns:
            ndarray int: an array of asynchronous system No. of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__intBuffer[index] = self.get_generator_sys_no(generator_no)
        return self.__intBuffer[:len(generator_list)].astype(np.int32)

    def get_generator_lf_bus_type(self, generator_no: int, sys_no=-1):
        """Get generator information: the power flow bus type of generator generator_no in asynchronous system sys_no.
           slack, pq, pv, pv_pq, pq_pv.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: power flow bus type of the generator.
        """        
        b_type = self.__psDLL.get_Generator_LF_Bus_Type(generator_no, sys_no)
        if b_type == 16:
            return 'slack'
        if b_type == 1:
            return 'pq'
        if b_type == 8:
            return 'pv'
        if b_type == 4:
            return 'pv_pq'
        if b_type == 2:
            return 'pq_pv'
        raise Exception("unknown lf bus type, please check!")

    def get_generator_all_lf_bus_type(self, generator_list=None, sys_no=-1):
        """Get generator information: all the power flow bus type of generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int: an array of power flow bus type of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        b_type = list()
        for generator_no in generator_list:
            b_type.append(self.get_generator_lf_bus_type(generator_no, sys_no))
        return np.array(b_type)
    
    def get_generator_all_ctrl(self, generator_list=None, sys_no=-1):
        """Get generator information: all the controllable generator No. in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int: an array of controllable generator No. in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        all_ctrl = np.arange(self.get_generator_number(None, sys_no)[self.get_generator_all_lf_bus_type(None, sys_no) != 'slack']) if sys_no != -1 else self.__indexCtrlGen
        return generator_list[np.where([gen in all_ctrl for gen in generator_list])]
    
    def get_generator_all_slack(self, generator_list=None, sys_no=-1):
        """Get generator information: all the slack generator No. in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray int: an array of slack generator No. in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        all_Slack = np.arange(self.get_generator_number(None, sys_no)[self.get_generator_all_lf_bus_type(None, sys_no) == 'slack']) if sys_no != -1 else self.__indexSlack
        return generator_list[np.where([gen in all_Slack for gen in generator_list])]

    def get_generator_v_set(self, generator_no: int, sys_no=-1):
        """Get generator information: the nodal voltage setting value of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: nodal voltage setting value of the generator.
        """        
        v_set = self.__psDLL.get_Generator_V0(generator_no, sys_no)
        assert v_set > -1.0e10, "generator v_set wrong, please check!"
        return v_set

    def get_generator_all_v_set(self, generator_list=None, sys_no=-1):
        """Get generator information: all the nodal voltage setting values of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of nodal voltage setting values of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_v_set(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def set_generator_v_set(self, vset: float, generator_no: int, sys_no=-1):
        """Set generator information: the nodal voltage setting value of generator generator_no in asynchronous system sys_no.

        Args:
            vset (float): nodal voltage setting value.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Generator_V0(vset, generator_no, sys_no), "set generator v set wrong, please check!"

    def set_generator_all_v_set(self, vset_array: np.ndarray, generator_list=None, sys_no=-1):
        """Set generator information: all the nodal voltage setting values of the generators in the generator_list in asynchronous system sys_no.
           Also see self.set_generator_v_set().

        Args:
            vset_array (np.ndarray): an array of nodal voltage setting values.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        assert len(vset_array) == len(generator_list), "generator number mismatch, please check!"
        for (vset, generator_no) in zip(vset_array, generator_list): self.set_generator_v_set(vset, generator_no, sys_no)

    def get_generator_p_set(self, generator_no: int, sys_no=-1):
        """Get generator information: the active power generation setting value of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: active power generation setting value of the generator.
        """        
        p_set = self.__psDLL.get_Generator_P0(generator_no, sys_no)
        assert p_set > -1.0e10, "generator p_set wrong, please check!"
        return p_set

    def get_generator_all_p_set(self, generator_list=None, sys_no=-1):
        """Get generator information: all the active power generation setting values of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of active power generation setting values of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_p_set(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def set_generator_p_set(self, pset: float, generator_no: int, sys_no=-1):
        """Set generator information: the active power generation setting value of generator generator_no in asynchronous system sys_no.

        Args:
            pset (float): active power generation setting value.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Generator_P0(pset, generator_no, sys_no), "set generator p set wrong, please check!"

    def set_generator_all_p_set(self, pset_array: np.ndarray, generator_list=None, sys_no=-1):
        """Set generator information: all the active power generation setting values of the generators in the generator_list in asynchronous system sys_no.
           Also see self.set_generator_v_set().

        Args:
            vset_array (np.ndarray): an array of active power generation setting values.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        assert len(pset_array) == len(generator_list), "generator number mismatch, please check!"
        for (pset, generator_no) in zip(pset_array, generator_list): self.set_generator_p_set(pset, generator_no, sys_no)

    def get_generator_vmax(self, generator_no: int, sys_no=-1):
        """Get generator information: the upper bound of nodal voltage of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the upper bound of nodal voltage of the generator.
        """        
        vmax = self.get_bus_vmax(self.get_generator_bus_no(generator_no, sys_no), sys_no)
        assert vmax > -1.0e10, "generator vmax wrong, please check!"
        return vmax

    def get_generator_all_vmax(self, generator_list=None, sys_no=-1):
        """Get generator information: all the upper bounds of nodal voltages of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of upper bounds of nodal voltage of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        return self.get_bus_all_vmax(None, sys_no)[self.get_generator_all_bus_no(generator_list, sys_no)]

    def get_generator_vmin(self, generator_no: int, sys_no=-1):
        """Get generator information: the lower bound of nodal voltage of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the lower bound of nodal voltage of the generator.
        """        
        vmin = self.get_bus_vmin(self.get_generator_bus_no(generator_no, sys_no), sys_no)
        assert vmin > -1.0e10, "generator vmin wrong, please check!"
        return vmin

    def get_generator_all_vmin(self, generator_list=None, sys_no=-1):
        """Get generator information: all the lower bounds of nodal voltages of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of lower bounds of nodal voltage setting values of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        return self.get_bus_all_vmin(None, sys_no)[self.get_generator_all_bus_no(generator_list, sys_no)]

    def get_generator_pmax(self, generator_no: int, sys_no=-1):
        """Get generator information: the upper bound of active power generation of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the upper bound of active power generation of the generator.
        """        
        pmax = self.__psDLL.get_Generator_PMax(generator_no, sys_no)
        assert pmax > -1.0e10, "generator pmax wrong, please check!"
        # pmax = self.get_generator_p_set(generator_no, sys_no) if abs(pmax) < 1.0e-6 else pmax
        return pmax

    def get_generator_all_pmax(self, generator_list=None, sys_no=-1):
        """Get generator information: all the upper bounds of active power generation of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of upper bounds of active power generation of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_pmax(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def get_generator_pmin(self, generator_no: int, sys_no=-1):
        """Get generator information: the lower bound of active power generation of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the lower bound of active power generation of the generator.
        """        
        pmin = self.__psDLL.get_Generator_PMin(generator_no, sys_no)
        assert pmin > -1.0e10, "generator pmin wrong, please check!"
        # pmin = self.get_generator_p_set(generator_no, sys_no) if abs(pmin) < 1.0e-6 else pmin
        return pmin

    # all generator pmin
    def get_generator_all_pmin(self, generator_list=None, sys_no=-1):
        """Get generator information: all the lower bounds of active power generation of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of lower bounds of active power generation of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_pmin(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def get_generator_qmax(self, generator_no: int, sys_no=-1):
        """Get generator information: the upper bound of reactive power generation of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the upper bound of reactive power generation of the generator.
        """        
        qmax = self.__psDLL.get_Generator_QMax(generator_no, sys_no)
        assert qmax > -1.0e10, "generator qmax wrong, please check!"
        return qmax

    def get_generator_all_qmax(self, generator_list=None, sys_no=-1):
        """Get generator information: all the upper bounds of reactive power generation of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of upper bounds of reactive power generation of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_qmax(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def get_generator_qmin(self, generator_no: int, sys_no=-1):
        """Get generator information: the lower bound of reactive power generation of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the lower bound of reactive power generation of the generator.
        """        
        qmin = self.__psDLL.get_Generator_QMin(generator_no, sys_no)
        assert qmin > -1.0e10, "generator qmin wrong, please check!"
        return qmin

    def get_generator_all_qmin(self, generator_list=None, sys_no=-1):
        """Get generator information: all the lower bounds of reactive power generation of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of lower bounds of reactive power generation of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_qmin(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def get_generator_tj(self, generator_no: int, sys_no=-1):
        """Get generator information: the innertia time constant of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: the innertia time constant of the generator.
        """        
        tj = self.__psDLL.get_Generator_Tj(generator_no, sys_no)
        assert tj > -1.0e10, "generator tj wrong, please check!"
        return tj

    def get_generator_all_tj(self, generator_list=None, sys_no=-1):
        """Get generator information: all the innertia time constants of the generators in the generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of innertia time constants of the generators in the generator_list.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.__doubleBuffer[index] = self.get_generator_tj(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    def get_generator_ts_type_name(self, generator_no: int, sys_no=-1):
        """Get generator information: the dynamic model type name of generator generator_no of asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            gbk str: the generator dynamic model name string.
        """        
        generator_ts_type_name = self.__psDLL.get_Generator_TS_Type(generator_no, sys_no)
        assert generator_no is not None,  "generator ts type name is empty, please check sys/generator no!"
        return string_at(generator_ts_type_name, -1).decode('gbk')

    def get_generator_all_ts_type_name(self, generator_list=None, sys_no=-1):
        """Get bus information: all the bus names of buses in the bus_list of asynchronous system sys_no.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray str: the array of generator ts type names.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        generator_ts_type_names = list()
        for generator_no in generator_list: generator_ts_type_names.append(self.get_bus_name(generator_no, sys_no))
        return np.array(generator_ts_type_names)

    def set_generator_environment_status(self, env_type: int, env_value: float, generator_no: int, sys_no=-1):
        """Set generator information: the environmental status of generator generator_no of asynchronous system sys_no.

        Args:
            env_type (int): the type of environmental change, 1 solar, 2 temperature, 3 wind.
            env_value (float): the environmental setting value.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Generator_Environment_Status(env_type, env_value, generator_no, sys_no) == True, "Generator environmental change failed, please check!"
    
    def set_generator_all_environment_status(self, env_type_list: list, env_value_list: list, generator_list=None, sys_no=-1):
        """Set generator information: all the environmental status of generators in the generator_list of asynchronous system sys_no.

        Args:
            env_type_list (list, int): the list of types of environmental changes, 1 solar, 2 temperature, 3 wind.
            env_value_list (list, float): the list of environmental setting values.
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        assert len(generator_list) == len(env_type_list) == len(env_value_list), f'Sizes do not match: generator_list ({len(generator_list)}), env_type_list ({len(env_type_list)}), and env_value_list ({len(env_value_list)})'
        for generator_no, env_type, env_value in zip(generator_list, env_type_list, env_value_list): self.set_generator_environment_status(env_type=env_type, env_value=env_value, generator_no=generator_no, sys_no=sys_no)

    def get_generator_lf_result(self, generator_no: int, sys_no=-1, buffer=None, rt=True):
        """Get generator information: the power flow result of generator generator_no of asychronous system sys_no.
           [0] active power generation
           [1] reactive power generation

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of bus power flow result. Array shape is (2,).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer            
        assert self.__psDLL.get_Generator_LF_Result(buffer, generator_no, sys_no) == True, "get lf generator result failed, please check!"
        if rt is True:
            return buffer[:2].astype(np.float32)

    def get_generator_all_lf_result(self, generator_list=None, sys_no=-1):
        """Get generator information: all the power flow results of generators in the generator_list of asychronous system sys_no.
           Also see self.get_generator_lf_result().

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of all the power flow results of generators. Array shape is (len(generator_list), 2). 
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): self.get_generator_lf_result(generator_no, sys_no, self.__doubleBuffer[index*2:], False)
        return self.__doubleBuffer[:len(generator_list)*2].reshape(len(generator_list), 2).astype(np.float32)

    def get_generator_ts_result_dimension(self, generator_no: int, sys_no=-1, need_inner_e=False):
        """Get generator information: the transient result dimension of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.

        Returns:
            int: transient result dimension. 
        """        
        dim = self.__psDLL.get_Generator_TS_Result_Dimension(generator_no, sys_no, need_inner_e)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim

    def get_generator_ts_cur_step_result(self, generator_no: int, sys_no=-1, need_inner_e=False, buffer=None, rt=True):
        """Get generator information: simulation result of generator generator_no of asynchronous system sys_no at current step. 
           The dimensionality can be accessed by self.get_generator_ts_result_dimension().
           [0] rotor angle, Deg.
           [1] rotation speed, Hz.
           [2] nodal voltage amplitude
           [3] nodal voltage phase
           [4] active power generation
           [5] reactive power generation
           [6-...] inner potential of generator
               classic model with constant E': [6] E'.
               2nd-order model with constant Eq': [6] Eq'.
               3nd-order model: [6] Eq'.
               4th-order model: [6] Eq', [7] Ed'.
               5th-order model: [6] Eq', [7] Ed'', [8] Eq''.
               6th-order model: [6] Eq', [7] Ed', [8] Ed'', [9] Eq''.
               solar model: [6] S, [7] T, [8] Vdc, [9] Ipv.
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_generator_ts_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of generator simulation result. Array shape is (self.get_generator_ts_result_dimension(),).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_TS_CurStep_Result(buffer, generator_no, sys_no, need_inner_e) == True, "get ts generator result failed, please check!"
        if rt is True:
            return buffer[:self.get_generator_ts_result_dimension(generator_no, sys_no, need_inner_e)].astype(np.float32)

    def get_generator_all_ts_cur_step_result(self, generator_list=None, sys_no=-1, need_inner_e=False):
        """Get generator information: all the simulation results of generators in generator_list of asynchronous system sys_no at current step. 
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_generator_all_ts_step_result().
           Currently, only the first 6 outputs can be accessed. TODO get all simulation results of different generator models.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.

        Returns:
            ndarray float32: an array of generator simulation results. Array shape is (len(generator_list), 6).
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list): 
            self.get_generator_ts_cur_step_result(generator_no, sys_no, need_inner_e, self.__doubleBuffer[index*6:], False)
        return self.__doubleBuffer[:len(generator_list)*6].reshape(len(generator_list), 6).astype(np.float32)

    def get_generator_ts_step_result(self, step: int, generator_no: int, sys_no=-1, need_inner_e=False):
        """Get generator information: simulation result of generator generator_no of asynchronous system sys_no at step. 
           The dimensionality can be accessed by self.get_generator_ts_result_dimension().
           Also see self.get_generator_ts_cur_step_result().

        Args:
            step (int): real integration step.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.

        Returns:
            ndarray float32: an array of generator simulation result. Array shape is (self.get_generator_ts_result_dimension(),).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_generator_ts_cur_step_result(generator_no, sys_no, need_inner_e)
    
    def get_generator_all_ts_step_result(self, step: int, generator_list=None, sys_no=-1, need_inner_e=False):
        """Get generator information: all the simulation results of generators in generator_list of asynchronous system sys_no at step. 
           Also see self.get_generator_all_ts_cur_step_result().
           Currently, only the first 6 outputs can be accessed. TODO get all simulation results of different generator models.

        Args:
            step (int): real integration step.
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.

        Returns:
            ndarray float32: an array of generator simulation results. Array shape is (len(generator_list), 6).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_generator_all_ts_cur_step_result(generator_list, sys_no, need_inner_e)

    def get_generator_ts_all_step_result(self, generator_no: int, sys_no=-1, need_inner_e=False):
        """Get generator information: all the simulation results of generator generator_no of asynchronous system sys_no. 
           Also see self.get_generator_ts_cur_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.

        Returns:
            ndarray float32: an array of generator simulation results. 
            Array shape is (self.get_info_ts_finish_step() + 1, self.get_generator_ts_result_dimension(generator_no, sys_no, need_inner_e)).
        """        
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_ts_result_dimension(generator_no, sys_no, need_inner_e)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no=sys_no)
            self.get_generator_ts_cur_step_result(generator_no, sys_no, need_inner_e, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    def get_generator_all_ts_result(self, generator_list=None, sys_no=-1, need_inner_e=False):
        """Get generator information: all the simulation results of generators in generator_list of asynchronous system sys_no. 
           Currently, only the first 6 outputs can be accessed. TODO get all simulation results of different generator models.

        Args:
            generator_list (list int, optional): the list of generators. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of generator. Defaults to False.

        Returns:
            ndarray float32: an array of generator simulation results. Array shape is (total_step, self.get_generator_number(sys_no), 6).
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Generator_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all generator results failed!"
        all_result = self.__doubleBuffer[0:total_step * 6 * self.get_generator_number(sys_no)].reshape(total_step, self.get_generator_number(sys_no), 6)
        return all_result[:, generator_list, :].astype(np.float32)
    
    ################################################################################################
    # exciter
    ################################################################################################
    def get_generator_exciter_ts_result_dimension(self, generator_no: int, sys_no=-1):
        """Get exciter information: the transient result dimension of exciter of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: transient result dimension. 
        """        
        dim = self.__psDLL.get_Generator_Exciter_TS_Result_Dimension(generator_no, sys_no)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim

    def get_generator_exciter_ts_cur_step_result(self, generator_no: int, sys_no=-1, buffer=None, rt=True):
        """Get exciter information: simulation result of exciter of generator generator_no of asynchronous system sys_no at current step. 
           The dimensionality can be accessed by self.get_generator_exciter_ts_result_dimension().
           [0] nodal voltage.
           ...
           [-1] excitation voltage Efd, p.u.
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_generator_exciter_ts_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of exciter simulation result. Array shape is (self.get_generator_exciter_ts_result_dimension(),).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_Exciter_TS_CurStep_Result(buffer, generator_no, sys_no) == True, "get ts generator exciter result failed, please check!"
        if rt is True:
            return buffer[:self.get_generator_exciter_ts_result_dimension(generator_no, sys_no)].astype(np.float32)

    def get_generator_exciter_ts_step_result(self, step: int, generator_no: int, sys_no=-1):
        """Get exciter information: simulation result of exciter of generator generator_no of asynchronous system sys_no at step. 
           The dimensionality can be accessed by self.get_generator_exciter_ts_result_dimension().
           Also see self.get_generator_exciter_ts_cur_step_result().

        Args:
            step (int): real integration step.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of exciter simulation result. Array shape is (self.get_generator_exciter_ts_result_dimension(),).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_generator_exciter_ts_cur_step_result(generator_no, sys_no)

    def get_generator_exciter_ts_all_step_result(self, generator_no: int, sys_no=-1):
        """Get exciter information: all the simulation result of exciter of generator generator_no of asynchronous system sys_no. 
           Also see self.get_generator_exciter_ts_cur_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of exciter simulation results. 
            Array shape is (self.get_info_ts_finish_step() + 1, self.get_generator_exciter_ts_result_dimension(generator_no, sys_no)).
        """        
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_exciter_ts_result_dimension(generator_no, sys_no)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no=sys_no)
            self.get_generator_exciter_ts_cur_step_result(generator_no, sys_no, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    ################################################################################################
    # governor
    ################################################################################################
    def get_generator_governor_ts_result_dimension(self, generator_no: int, sys_no=-1):
        """Get governor information: the transient result dimension of governor of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: transient result dimension. 
        """        
        dim = self.__psDLL.get_Generator_Governor_TS_Result_Dimension(generator_no, sys_no)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim
    
    def get_generator_governor_ts_cur_step_result(self, generator_no: int, sys_no=-1, buffer=None, rt=True):
        """Get governor information: simulation result of governor of generator generator_no of asynchronous system sys_no at current step. 
           The dimensionality can be accessed by self.get_generator_governor_ts_result_dimension().
           [0] rotation speed, Hz.
           ...
           [-1] mechnical power output, p.u.
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_generator_governor_ts_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of governor simulation result. Array shape is (self.get_generator_governor_ts_result_dimension(),).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_Governor_TS_CurStep_Result(buffer, generator_no, sys_no) == True, "get ts generator governor result failed, please check!"
        if rt is True:
            return buffer[:self.get_generator_governor_ts_result_dimension(generator_no, sys_no)].astype(np.float32)

    def get_generator_governor_ts_step_result(self, step: int, generator_no: int, sys_no=-1):
        """Get governor information: simulation result of governor of generator generator_no of asynchronous system sys_no at step. 
           The dimensionality can be accessed by self.get_generator_governor_ts_result_dimension().
           Also see self.get_generator_governor_ts_cur_step_result().

        Args:
            step (int): real integration step.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of governor simulation result. Array shape is (self.get_generator_governor_ts_result_dimension(),).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_generator_governor_ts_cur_step_result(generator_no, sys_no)

    def get_generator_governor_ts_all_step_result(self, generator_no: int, sys_no=-1):
        """Get governor information: all the simulation result of governor of generator generator_no of asynchronous system sys_no. 
           Also see self.get_generator_governor_ts_cur_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of governor simulation results. 
            Array shape is (self.get_info_ts_finish_step() + 1, self.get_generator_governor_ts_result_dimension(generator_no, sys_no)).
        """        
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_governor_ts_result_dimension(generator_no, sys_no)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no=sys_no)
            self.get_generator_governor_ts_cur_step_result(generator_no, sys_no, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    ################################################################################################
    # pss
    ################################################################################################
    def get_generator_pss_ts_result_dimension(self, generator_no: int, sys_no=-1):
        """Get pss information: the transient result dimension of pss of generator generator_no in asynchronous system sys_no.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: transient result dimension. 
        """        
        dim = self.__psDLL.get_Generator_PSS_TS_Result_Dimension(generator_no, sys_no)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim

    def get_generator_pss_ts_cur_step_result(self, generator_no: int, sys_no=-1, buffer=None, rt=True):
        """Get pss information: simulation result of pss of generator generator_no of asynchronous system sys_no at current step. 
           The dimensionality can be accessed by self.get_generator_pss_ts_result_dimension().
           [0] rotation speed, Hz.
           ...
           [-1] mechnical power output, p.u.
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_generator_pss_ts_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of pss simulation result. Array shape is (self.get_generator_pss_ts_result_dimension(),).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_PSS_TS_CurStep_Result(buffer, generator_no, sys_no) == True, "get ts generator pss result failed, please check!"
        if rt is True:
            buffer[:self.get_generator_pss_ts_result_dimension(generator_no, sys_no)].astype(np.float32)

    def get_generator_pss_ts_step_result(self, step: int, generator_no: int, sys_no=-1):
        """Get pss information: simulation result of pss of generator generator_no of asynchronous system sys_no at step. 
           The dimensionality can be accessed by self.get_generator_pss_ts_result_dimension().
           Also see self.get_generator_pss_ts_cur_step_result().

        Args:
            step (int): real integration step.
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of pss simulation result. Array shape is (self.get_generator_pss_ts_result_dimension(),).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_generator_pss_ts_cur_step_result(generator_no, sys_no)

    def get_generator_pss_ts_all_step_result(self, generator_no: int, sys_no=-1):
        """Get pss information: all the simulation result of pss of generator generator_no of asynchronous system sys_no. 
           Also see self.get_generator_pss_ts_cur_step_result().

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of pss simulation results. 
            Array shape is (self.get_info_ts_finish_step() + 1, self.get_generator_pss_ts_result_dimension(generator_no, sys_no)).
        """        
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_pss_ts_result_dimension(generator_no, sys_no)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no=sys_no)
            self.get_generator_pss_ts_cur_step_result(generator_no, sys_no, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    ################################################################################################
    # load
    ################################################################################################
    def get_load_number(self, sys_no=-1):
        """Get load information: the number of loads in asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the number of loads.
        """        
        if sys_no == -1:
            return self.__nLoad
        n_load = self.__psDLL.get_Load_Number(sys_no)
        assert n_load >= 0, 'total number of load wrong, please check!'
        return n_load

    def get_load_bus_no(self, load_no: int, sys_no=-1):
        """Get load information: the bus No. of load load_no in asynchronous system sys_no.

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: bus No. of the load.
        """        
        bus_no = self.__psDLL.get_Load_Bus_No(load_no, sys_no)
        assert bus_no >= 0, "load i no wrong, please check!"
        return bus_no

    def get_load_all_bus_no(self, load_list=None, sys_no=-1):
        """Get load information: all the bus No. of loads in the load_list in asynchronous system sys_no.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.

        Returns:
            ndarray int: an array of the bus No. of the loads in the load_list.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__intBuffer[index] = self.get_load_bus_no(load_no, sys_no)
        return self.__intBuffer[:len(load_list)].astype(np.int32)

    def get_load_bus_name(self, load_no: int, sys_no=-1):
        """Get load information: the bus name of load load_no in asynchronous system sys_no.

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            str: bus name of the load.
        """        
        return self.get_bus_name(self.get_load_bus_no(load_no, sys_no), sys_no)

    def get_load_all_bus_name(self, load_list=None, sys_no=-1):
        """Get load information: all the bus names of loads in the load_list in asynchronous system sys_no.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.

        Returns:
            ndarray str: an array of the bus names of the loads in the load_list.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        return self.get_bus_all_name(None, sys_no)[self.get_load_all_bus_no(load_list, sys_no)]

    def get_load_sys_no(self, load_no: int):
        """Get load information: the asynchronous system No. of load load_no.

        Args:
            load_no (int): load No. of asynchronous system sys_no.

        Returns:
            int: asynchronous system No. of the load.
        """        
        sys_no = self.__psDLL.get_Load_Sys_No(load_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    def get_load_all_sys_no(self, load_list=None):
        """Get load information: all the asynchronous system No. of loads in the load_list.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.

        Returns:
            ndarray int: an array of asynchronous system No. of the loads in the load_list.
        """        
        load_list = np.arange(self.get_load_number(), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__intBuffer[index] = self.get_load_sys_no(load_no)
        return self.__intBuffer[:len(load_list)].astype(np.int32)

    def get_load_p_set(self, load_no: int, sys_no=-1):
        """Get load information: the active power load setting value of load load_no in asynchronous system sys_no.

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: active power load setting value of the load.
        """        
        v_set = self.__psDLL.get_Load_P0(load_no, sys_no)
        assert v_set > -1.0e10, "load p_set wrong, please check!"
        return v_set

    def get_load_all_p_set(self, load_list=None, sys_no=-1):
        """Get load information: all the active power load setting values of the loads in the load_list in asynchronous system sys_no.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of active power load setting values of the loads in the load_list.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__doubleBuffer[index] = self.get_load_p_set(load_no, sys_no)
        return self.__doubleBuffer[:len(load_list)].astype(np.float32)

    def set_load_p_set(self, pset: float, load_no: int, sys_no=-1):
        """Set load information: the active power load setting value of load load_no in asynchronous system sys_no.

        Args:
            pset (float): active power load setting value.
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Load_P0(pset, load_no, sys_no), "set load p set wrong, please check!"

    # set all load p set
    def set_load_all_p_set(self, pset_array: np.ndarray, load_list=None, sys_no=-1):
        """Set load information: all the active power load setting values of the loads in the load_list in asynchronous system sys_no.
           Also see self.set_load_v_set().

        Args:
            qset_array (np.ndarray): an array of active power load setting values.
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        assert len(pset_array) == len(load_list), "load number mismatch, please check!"
        for (pset, load_no) in zip(pset_array, load_list): self.set_load_p_set(pset, load_no, sys_no)

    def get_load_q_set(self, load_no: int, sys_no=-1):
        """Get load information: the re power load setting value of load load_no in asynchronous system sys_no.

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            float: reactive power load setting value of the load.
        """        
        q_set = self.__psDLL.get_Load_Q0(load_no, sys_no)
        assert q_set > -1.0e10, "load q_set wrong, please check!"
        return q_set

    def get_load_all_q_set(self, load_list=None, sys_no=-1):
        """Get load information: all the reactive power load setting values of the loads in the load_list in asynchronous system sys_no.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of reactive power load setting values of the loads in the load_list.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__doubleBuffer[index] = self.get_load_q_set(load_no, sys_no)
        return self.__doubleBuffer[:len(load_list)].astype(np.float32)

    def set_load_q_set(self, qset: float, load_no: int, sys_no=-1):
        """Set load information: the reactive power load setting value of load load_no in asynchronous system sys_no.

        Args:
            qset (float): reactive power load setting value.
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Load_Q0(qset, load_no, sys_no), "set load p set wrong, please check!"

    def set_load_all_q_set(self, qset_array: np.ndarray, load_list=None, sys_no=-1):
        """Set load information: all the reactive power load setting values of the loads in the load_list in asynchronous system sys_no.
           Also see self.set_load_v_set().

        Args:
            qset_array (np.ndarray): an array of reactive power load setting values.
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        assert len(qset_array) == len(load_list), "load number mismatch, please check!"
        for (qset, load_no) in zip(qset_array, load_list): self.set_load_q_set(qset, load_no, sys_no)

    def get_load_lf_result(self, load_no: int, sys_no=-1, buffer=None, rt=True):
        """Get load information: the power flow result of load load_no of asychronous system sys_no.
           [0] active power load
           [1] reactive power load

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of bus power flow result. Array shape is (2,).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Load_LF_Result(buffer, load_no, sys_no) == True, "get lf load result failed, please check!"
        if rt is True:
            return buffer[:2].astype(np.float32)

    def get_load_all_lf_result(self, load_list=None, sys_no=-1):
        """Get load information: all the power flow results of loads in the load_list of asychronous system sys_no.
           Also see self.get_load_lf_result().

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of all the power flow results of loads. Array shape is (len(load_list), 2). 
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.get_load_lf_result(load_no, sys_no, self.__doubleBuffer[index*2:], False)
        return self.__doubleBuffer[:len(load_list)*2].reshape(len(load_list), 2).astype(np.float32)

    def get_load_ts_result_dimension(self, load_no: int, sys_no=-1, need_dynamic_variable=False):
        """Get load information: the transient result dimension of load load_no in asynchronous system sys_no.

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_dynamic_variable (bool, optional): the flag shows whether considering dynamic state variables of load. Defaults to False.

        Returns:
            int: transient result dimension. 
        """        
        dim = self.__psDLL.get_Load_TS_Result_Dimension(load_no, sys_no, need_dynamic_variable)
        assert dim >= 0, 'load i no wrong, please check!'
        return dim

    def get_load_ts_cur_step_result(self, load_no: int, sys_no=-1, need_dynamic_variable=False, buffer=None, rt=True):
        """Get load information: simulation result of load load_no of asynchronous system sys_no at current step. 
           The dimensionality can be accessed by self.get_load_ts_result_dimension().
           [0] nodal voltage amplitude.
           [1] nodal voltage phase.
           [2] active power load.
           [3] reactive power load.
           [4...] dynamic state variables.
               induction motor: [4] slip frequency, [5] real part of the inner potential, [6] imaginary part of the inner potential, [7] mechanical torque.
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_load_ts_step_result().

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_dynamic_variable (bool, optional): the flag shows whether considering dynamic state variables of load. Defaults to False.
            buffer (ndarray float64, optional): The buffer used to save data. Defaults to None, which means the buffer created in self._create_buffer() will be used.
            rt (bool, optional): a flag showing whether returning the array back after loading the data to the buffer. Defaults to True.

        Returns:
            ndarray float32: an array of load simulation result. Array shape is (self.get_load_ts_result_dimension(),).
        """        
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Load_TS_CurStep_Result(buffer, load_no, sys_no, need_dynamic_variable) == True, "get ts load result failed, please check!"
        if rt is True:
            return buffer[:self.get_load_ts_result_dimension(load_no, sys_no, need_dynamic_variable)].astype(np.float32)

    def get_load_all_ts_cur_step_result(self, load_list=None, sys_no=-1, need_dynamic_variable=False):
        """Get load information: all the simulation results of loads in load_list of asynchronous system sys_no at current step. 
           The funtion is usually used with self.set_info_ts_step_element_state(). Also see self.get_load_all_ts_step_result().
           Currently, only the first 6 outputs can be accessed. TODO get all simulation results of different load models.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_dynamic_variable (bool, optional): the flag shows whether considering dynamic state variables of load. Defaults to False.

        Returns:
            ndarray float32: an array of load simulation results. Array shape is (len(load_list), 4).
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.get_load_ts_cur_step_result(load_no, sys_no, need_dynamic_variable, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(load_list)*4].reshape(len(load_list), 4).astype(np.float32)

    def get_load_ts_step_result(self, step: int, load_no: int, sys_no=-1, need_dynamic_variable=False):
        """Get load information: simulation result of load load_no of asynchronous system sys_no at step. 
           The dimensionality can be accessed by self.get_load_ts_result_dimension().
           Also see self.get_load_ts_cur_step_result().

        Args:
            step (int): real integration step.
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_dynamic_variable (bool, optional): the flag shows whether considering dynamic state variables of load. Defaults to False.

        Returns:
            ndarray float32: an array of load simulation result. Array shape is (self.get_load_ts_result_dimension(),).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_load_ts_cur_step_result(load_no, sys_no, need_dynamic_variable)

    def get_load_all_ts_step_result(self, step: int, load_list=None, sys_no=-1, need_dynamic_variable=False):
        """Get load information: all the simulation results of loads in load_list of asynchronous system sys_no at step. 
           Also see self.get_load_all_ts_cur_step_result().
           Currently, only the first 6 outputs can be accessed. TODO get all simulation results of different load models.

        Args:
            step (int): real integration step.
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of load. Defaults to False.

        Returns:
            ndarray float32: an array of load simulation results. Array shape is (len(load_list), 4).
        """        
        self.set_info_ts_step_element_state(step, sys_no=sys_no)
        return self.get_load_all_ts_cur_step_result(load_list, sys_no, need_dynamic_variable)

    def get_load_ts_all_step_result(self, load_no: int, sys_no=-1, need_dynamic_variable=False):
        """Get load information: all the simulation results of load load_no of asynchronous system sys_no. 
           Also see self.get_load_ts_cur_step_result().

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of load. Defaults to False.

        Returns:
            ndarray float32: an array of load simulation results. 
            Array shape is (self.get_info_ts_finish_step() + 1, self.get_load_ts_result_dimension(load_no, sys_no, need_dynamic_variable)).
        """        
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_load_ts_result_dimension(load_no, sys_no, need_dynamic_variable)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no=sys_no)
            self.get_load_ts_cur_step_result(load_no, sys_no, need_dynamic_variable, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    def get_load_all_ts_result(self, load_list=None, sys_no=-1, need_dynamic_variable=False):
        """Get load information: all the simulation results of loads in load_list of asynchronous system sys_no. 
           Currently, only the first 6 outputs can be accessed. TODO get all simulation results of different load models.

        Args:
            load_list (list int, optional): the list of loads. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
            need_inner_e (bool, optional): the flag shows whether considering the inner potential of load. Defaults to False.

        Returns:
            ndarray float32: an array of load simulation results. Array shape is (total_step, self.get_load_number(sys_no), 4).
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Load_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all load results failed!"
        all_result = self.__doubleBuffer[0:total_step * 4 * self.get_load_number(sys_no)].reshape(total_step, self.get_load_number(sys_no), 4)
        return all_result[:, load_list, :].astype(np.float32)

    ################################################################################################
    # network
    ################################################################################################
    def get_network_n_non_zero(self, sys_no=-1):
        """Get network information: number of non-zero elements in the factor table of asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: number of non-zero elements in the factor table.
        """        
        if sys_no == -1:
            return self.__nNonzero
        n_non = self.__psDLL.get_Network_N_Non_Zero_Element(sys_no)
        assert n_non >= 0, 'total number of non-zero element wrong, please check!'
        return n_non

    def get_network_n_inverse_non_zero(self, sys_no=-1):
        """Get network information: number of non-zero elements in the inverse factor table of asynchronous system sys_no.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: number of non-zero elements in the factor table.
        """        
        if sys_no == -1: return self.__nInverseNonZero
        n_non = self.__psDLL.get_Network_N_Inverse_Non_Zero_Element(sys_no)
        assert n_non >= 0, 'total number of inverse non-zero element wrong, please check!'
        return n_non

    def get_network_n_acsystem_check_connectivity(self, ts_step=0, sys_no=-1):
        """Get network information: check network connectivity and get the number of asynchronous systems at integration step ts_step in asynchronous system sys_no.
           For example, ts_step=0 means the connectivity of the stable operation state, ts_step=20 means the connectivity at integration step 20.

        Args:
            ts_step (int, optional): simulation step. Defaults to 0.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            int: the number of asynchronous systems.
        """        
        self.set_info_ts_step_element_state(ts_step)
        n_acsystem = self.__psDLL.get_Network_N_ACSystem_Check_Connectivity(sys_no)
        assert n_acsystem >= 0, "ac system no. is not correct, please check!"
        return n_acsystem

    def get_network_bus_connectivity_flag(self, bus_no: int, sys_no=-1):
        """Get network information: get connectivity flag of bus bus_no.

        Args:
            bus_no (int): Bus No.
            sys_no (int, optional): System No. Defaults to -1, meaning the whole system network.

        Returns:
            int: Bus connectivity flag, e.g., if bus 1 belongs to asynchronous subsystem 2 then return 2. 
        """        
        return self.__psDLL.get_Network_Bus_Connectivity_Flag(bus_no, sys_no)

    def get_network_bus_all_connectivity_flag(self, bus_list=None, sys_no=-1):
        """Get network information: all the asynchronous subsystem NO. of buses in the bus_list.

        Args:
            bus_list (list int, optional): the list of buses whose asynchronous subsystem No. is needed. Defaults to None, which means all the buses.

        Returns:
            ndarray int: an array of the asynchronous system No.
        """        
        bus_list = np.arange(self.__nBus, dtype=np.int32) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__intBuffer[index] = self.get_network_bus_connectivity_flag(bus_no, sys_no)
        return self.__intBuffer[:len(bus_list)].astype(np.int32)

    def get_network_acline_connectivity(self, acline_no: int, sys_no=-1):
        """Get network information: the connectivity of acline acline_no in asynchronous system sys_no.
           The connectivity of the acline is True if it is connected to the power network, otherwise the connectivity will be False.

        Args:
            acline_no (int): acline No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            bool: acline connectivity.
        """        
        return self.__psDLL.get_Network_ACLine_Connectivity(acline_no, sys_no)

    def get_network_acline_all_connectivity(self, acline_list=None, sys_no=-1):
        """Get network information: all the connectivity of aclines in acline_list in asynchronous system sys_no.

        Args:
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray bool: an array of acline connectivity.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)),acline_list):
            self.__boolBuffer[index] = self.get_network_acline_connectivity(acline_no, sys_no)
        return self.__boolBuffer[:len(acline_list)].astype(bool)

    def set_network_acline_connectivity(self, cmark: bool, acline_no: int, sys_no=-1):
        """Set network information: the connectivity of acline acline_no in asynchronous system sys_no.

        Args:
            cmark (bool): connectivity flag. 
            acline_no (int): acline No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Network_ACLine_Connectivity(cmark, acline_no, sys_no), "set acline connectivity mark wrong, please check!"

    def set_network_acline_all_connectivity(self, cmarks: np.ndarray, acline_list=None, sys_no=-1):
        """Set network information: all the connectivity of aclines in acline_list in asynchronous system sys_no.

        Args:
            cmarks (np.ndarray): an array of connectivity flags. 
            acline_list (list int, optional): list of acline No. Defaults to None, which means all the aclines.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int32) if acline_list is None else acline_list
        assert len(acline_list) == len(cmarks), "marks length does not match, please cleck"
        for (cmark, acline_no) in zip(cmarks, acline_list): self.set_network_acline_connectivity(cmark, acline_no, sys_no)

    def get_network_transformer_connectivity(self, transformer_no: int, sys_no=-1):
        """Get network information: the connectivity of transformer transformer_no in asynchronous system sys_no.
           The connectivity of the transformer is True if it is connected to the power network, otherwise the connectivity will be False.

        Args:
            transformer_no (int): transformer No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            bool: transformer connectivity.
        """        
        return self.__psDLL.get_Network_Transformer_Connectivity(transformer_no, sys_no)

    def get_network_transformer_all_connectivity(self, transformer_list=None, sys_no=-1):
        """Get network information: all the connectivity of transformers in transformer_list in asynchronous system sys_no.

        Args:
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray bool: an array of transformer connectivity.
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__boolBuffer[index] = self.get_network_transformer_connectivity(transformer_no, sys_no)
        return self.__boolBuffer[:len(transformer_list)].astype(bool)

    def set_network_transformer_connectivity(self, cmark: bool, transformer_no: int, sys_no=-1):
        """Set network information: the connectivity of transformer transformer_no in asynchronous system sys_no.

        Args:
            cmark (bool): connectivity flag. 
            transformer_no (int): transformer No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Network_Transformer_Connectivity(cmark, transformer_no, sys_no), "set transformer connectivity mark wrong, please check!"

    def set_network_transformer_all_connectivity(self, cmarks: np.ndarray, transformer_list=None, sys_no=-1):
        """Set network information: all the connectivity of transformers in transformer_list in asynchronous system sys_no.

        Args:
            cmarks (np.ndarray): an array of connectivity flags. 
            transformer_list (list int, optional): list of transformer No. Defaults to None, which means all the transformers.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int32) if transformer_list is None else transformer_list
        assert len(transformer_list) == len(cmarks), "marks length does not match, please cleck"
        for (cmark, transformer_no) in zip(cmarks, transformer_list): self.set_network_transformer_connectivity(cmark, transformer_no, sys_no)

    def set_network_rebuild_all_network_data(self):
        """Set network information: rebuild all network data, including admittance matrix, factor table, sparse vector path, and path tree.
        """        
        assert self.__psDLL.set_Network_Rebuild_All_Network_Data() is True, "rebuild network data failed, please check"

    def get_network_generator_connectivity(self, generator_no: int, sys_no=-1):
        """Get network information: the connectivity of generator generator_no in asynchronous system sys_no.
           The connectivity of the generator is True if it is connected to the power network, otherwise the connectivity will be False.

        Args:
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            bool: generator connectivity.
        """        
        return self.__psDLL.get_Network_Generator_Connectivity(generator_no, sys_no)

    def get_network_generator_all_connectivity(self, generator_list=None, sys_no=-1):
        """Get network information: all the connectivity of generators in generator_list in asynchronous system sys_no.

        Args:
            generator_list (list int, optional): list of generator No. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray bool: an array of generator connectivity.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__boolBuffer[index] = self.get_network_generator_connectivity(generator_no, sys_no)
        return self.__boolBuffer[:len(generator_list)].astype(bool)

    def set_network_generator_connectivity(self, cmark: bool, generator_no: int, sys_no=-1):
        """Set network information: the connectivity of generator generator_no in asynchronous system sys_no.

        Args:
            cmark (bool): connectivity flag. 
            generator_no (int): generator No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Network_Generator_Connectivity(cmark, generator_no, sys_no), "set generator connectivity mark wrong, please check!"

    def set_network_generator_all_connectivity(self, cmarks: np.ndarray, generator_list=None, sys_no=-1):
        """Set network information: all the connectivity of generators in generator_list in asynchronous system sys_no.

        Args:
            cmarks (np.ndarray): an array of connectivity flags. 
            generator_list (list int, optional): list of generator No. Defaults to None, which means all the generators.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int32) if generator_list is None else generator_list
        assert len(generator_list) == len(cmarks), "marks length does not match, please cleck!"
        for (cmark, generator_no) in zip(cmarks, generator_list): self.set_network_generator_connectivity(cmark, generator_no, sys_no)
    
    def get_network_load_connectivity(self, load_no: int, sys_no=-1):
        """Get network information: the connectivity of load load_no in asynchronous system sys_no.
           The connectivity of the load is True if it is connected to the power network, otherwise the connectivity will be False.

        Args:
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            bool: load connectivity.
        """        
        return self.__psDLL.get_Network_Load_Connectivity(load_no, sys_no)

    def get_network_load_all_connectivity(self, load_list=None, sys_no=-1):
        """Get network information: all the connectivity of loads in load_list in asynchronous system sys_no.

        Args:
            load_list (list int, optional): list of load No. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray bool: an array of load connectivity.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__boolBuffer[index] = self.get_network_load_connectivity(load_no, sys_no)
        return self.__boolBuffer[:len(load_list)].astype(bool)

    def set_network_load_connectivity(self, cmark: bool, load_no: int, sys_no=-1):
        """Set network information: the connectivity of load load_no in asynchronous system sys_no.

        Args:
            cmark (bool): connectivity flag. 
            load_no (int): load No. of asynchronous system sys_no.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.__psDLL.set_Network_Load_Connectivity(cmark, load_no, sys_no), "set load connectivity mark wrong, please check!"

    def set_network_load_all_connectivity(self, cmarks: np.ndarray, load_list=None, sys_no=-1):
        """Set network information: all the connectivity of loads in load_list in asynchronous system sys_no.

        Args:
            cmarks (np.ndarray): an array of connectivity flags. 
            load_list (list int, optional): list of load No. Defaults to None, which means all the loads.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int32) if load_list is None else load_list
        assert len(load_list) == len(cmarks), "marks length does not match, please cleck!"
        for (cmark, load_no) in zip(cmarks, load_list): self.set_network_load_connectivity(cmark, load_no, sys_no)

    def set_network_topology_original(self, sys_no=-1):
        """Integrated function: set the topology back to the original one.

        Args:
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        acline_cmarks = np.full(self.get_acline_number(sys_no), True)
        self.set_network_acline_all_connectivity(acline_cmarks, None, sys_no)
        transformer_cmarks = np.full(self.get_transformer_number(sys_no), True)
        self.set_network_transformer_all_connectivity(transformer_cmarks, None, sys_no)
        generator_cmarks = np.full(self.get_generator_number(sys_no), True)
        self.set_network_generator_all_connectivity(generator_cmarks, None, sys_no)
        load_cmarks = np.full(self.get_load_number(sys_no), True)
        self.set_network_load_all_connectivity(load_cmarks, None, sys_no)

    def get_network_admittance_matrix_full(self, ts_step=0, is_real_step=True, sys_no=-1):
        """Get network information: a full matrix of the admittance matrix at integration step ts_step in asynchronous system sys_no. 
           For example, ts_step=0 means the admittance matrix of the stable operation state, ts_step=20 means the admittance matrix at integration step 20.
           If is_real_step is true, the step is the real integration step, e.g., when integration step is 0.01s, t=0.0s is step 0, t=0.15 is step 15, t=2.34s is step 234, etc.
           If is_real_step is false, the step is the actual step in the simulator PSOPS. 

        Args:
            ts_step (int, optional): simulation step. Defaults to 0.
            is_real_step (bool, optional): A flag showing whether the step is a real one or the actual step in simulation. Defaults to True.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of admittance matrix. Array shape is (2, n_bus, n_bus). (0, n_bus, n_bus) is the real part. (1, n_bus, n_bus) is the imaginary part.
        """        
        self.set_info_ts_step_network_state(ts_step, is_real_step, sys_no)
        assert self.__psDLL.get_Network_Admittance_Matrix_Full(self.__doubleBuffer, sys_no), "get full admittance matrix failed, please check!"
        n_bus = self.get_bus_number(sys_no)
        return self.__doubleBuffer[:2*n_bus*n_bus].reshape(2, n_bus, n_bus).astype(np.float32)

    def get_network_impedance_matrix_full(self, ts_step=0, is_real_step=True, sys_no=-1):
        """Get network information: a full matrix of the impedance matrix, which is naturally a full matrix, at integration step ts_step in asynchronous system sys_no.
           For example, ts_step=0 means the admittance matrix of the stable operation state, ts_step=20 means the admittance matrix at integration step 20.
           If is_real_step is true, the step is the real integration step, e.g., when integration step is 0.01s, t=0.0s is step 0, t=0.15 is step 15, t=2.34s is step 234, etc.
           If is_real_step is false, the step is the actual step in the simulator PSOPS. 

        Args:
            ts_step (int, optional): simulation step. Defaults to 0.
            is_real_step (bool, optional): A flag showing whether the step is a real one or the actual step in simulation. Defaults to True.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of impedance matrix. Array shape is (2, n_bus, n_bus). (0, n_bus, n_bus) is the real part. (1, n_bus, n_bus) is the imaginary part.
        """        
        self.set_info_ts_step_network_state(ts_step, is_real_step, sys_no)
        assert self.__psDLL.get_Network_Impedence_Matrix_Full(self.__doubleBuffer, sys_no), "get impedance matrix failed, please check!"
        n_bus = self.get_bus_number(sys_no)
        return self.__doubleBuffer[:2*n_bus*n_bus].reshape(2, n_bus, n_bus).astype(np.float32)

    def get_network_impedance_matrix_factorized(self, ts_step=0, is_real_step=True, sys_no=-1):
        """Get network information: a factorized matrix of the impedance matrix at integration step ts_step in asynchronous system sys_no.
           For example, ts_step=0 means the admittance matrix of the stable operation state, ts_step=20 means the admittance matrix at integration step 20.
           If is_real_step is true, the step is the real integration step, e.g., when integration step is 0.01s, t=0.0s is step 0, t=0.15 is step 15, t=2.34s is step 234, etc.
           If is_real_step is false, the step is the actual step in the simulator PSOPS. 

        Args:
            ts_step (int, optional): simulation step. Defaults to 0.
            is_real_step (bool, optional): A flag showing whether the step is a real one or the actual step in simulation. Defaults to True.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            ndarray float32: an array of the factorized impedance matrix. Array shape is (3,). 
            The first one is the upper triangle matrix of the factorized impedance matrix, the array shape is (n_invnonzero+n_bus, 6).
            The second one is the lower triangle matrix of the factorized impedance matrix, the array shape is (n_invnonzero+n_bus, 6).
            The third one is the diagonal matrix of of the factorized impedance matrix, the array shape is (n_bus, 6).
            Each row is (pos_i, pos_j, (b, g, g, -b)).
        """        
        self.set_info_ts_step_network_state(ts_step, is_real_step, sys_no)
        assert self.__psDLL.get_Network_Impedence_Matrix_Factorized(self.__doubleBuffer, sys_no), "get factorized inverse matrix failed, please check!"
        n_bus = self.get_bus_number(sys_no)
        n_invnonzero = self.get_network_n_inverse_non_zero(sys_no)
        f_inv = list()
        f_inv.append(self.__doubleBuffer[0:(n_invnonzero+n_bus)*6].reshape(n_invnonzero+n_bus, 6).astype(np.float32))
        f_inv.append(self.__doubleBuffer[(n_invnonzero+n_bus)*6:2*(n_invnonzero+n_bus)*6].reshape(n_invnonzero+n_bus, 6).astype(np.float32))
        f_inv.append(self.__doubleBuffer[2*(n_invnonzero+n_bus)*6:2*(n_invnonzero+n_bus)*6+n_bus*6].reshape(n_bus, 6).astype(np.float32))
        return np.array(f_inv, dtype=object)

    def get_network_topology_sample(self, topo_change=1, sys_no=-1):
        """Integrated function: get an new topology with topo_change randomly chosen aclines cutting out.
           Connectivity must be kept.

        Args:
            topo_change (int, optional): the number of the aclines cutting out. Defaults to 1.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            list: information of the aclines cutting out.
        """        
        self.set_network_topology_original()
        if topo_change == 0: 
            self.set_network_rebuild_all_network_data()
            return None
        acline_no = np.arange(self.get_acline_number(sys_no))
        # acline_no = np.array([1,2,3,4,7,8,9,10,11,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])
        n_sample = 0
        selected_no = None
        while True:
            selected_no = self.__rng.choice(acline_no, size=topo_change, replace=False)
            for line_no in selected_no:
                self.set_network_acline_connectivity(False, line_no, sys_no)
            if self.get_network_n_acsystem_check_connectivity() == self.__nACSystem:
                break
            for line_no in selected_no:
                self.set_network_acline_connectivity(True, line_no, sys_no)
            n_sample += 1
            assert n_sample < 100, "topology sample failed, please check!"
        self.set_network_rebuild_all_network_data()
        # print([[line_no, self.get_acline_info(line_no, sys_no)] for line_no in selected_no])
        return [[line_no, self.get_acline_info(line_no, sys_no)] for line_no in selected_no]

    ################################################################################################
    # fault and disturbance
    ################################################################################################
    def set_fault_disturbance_clear_all(self):
        """Set fault and disturbance information: clear all the faults and disturbances.
        """        
        assert self.__psDLL.set_Fault_Disturbance_Clear_All() is True, "clear fault and disturbance failed, please check!"

    def set_fault(self, fault_type: int, fault_dis: float, start_time: float, end_time: float, ele_type: int, ele_pos: int, mod_flg: bool, sys_no=-1):
        """Set fault and disturbance information: add an acline fault according to settings in asynchronous system sys_no.

        Args:
            fault_type (int): fault type, 0-three phase fault, 1-three phase disconnection
            fault_dis (float): fault distance, 0-100.
            start_time (float): starting time. The fault happens at this time.
            end_time (float): end time. The fault ends at this time.
            ele_type (int): component type, 0-acline, 1-transformer.
            ele_pos (int): acline No. of the fault acline.
            mod_flg (bool): flag of modification, False-add new disturbance, True-modify exsiting disturbance.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        
        Returns:
            bool: whether the acline is set successfully.
        """        
        return self.__psDLL.set_Fault_Disturbance_Add_Fault(fault_type, fault_dis, start_time, end_time, ele_type, ele_pos, mod_flg, sys_no)

    def set_fault_disturbance_add_acline(self, fault_type: int, fault_dis: float, start_time: float, end_time: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: add an acline fault according to settings in asynchronous system sys_no.

        Args:
            fault_type (int): fault type, 0-three phase fault, 1-three phase disconnection
            fault_dis (float): fault distance, 0-100.
            start_time (float): starting time. The fault happens at this time.
            end_time (float): end time. The fault ends at this time.
            ele_pos (int): acline No. of the fault acline.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_fault(fault_type, fault_dis, start_time, end_time, 0, ele_pos, False, sys_no), \
            f'add fault acline failed, {[fault_type, fault_dis, start_time, end_time, ele_pos, sys_no]}, please check!'

    def set_fault_disturbance_change_acline(self, fault_type: int, fault_dis: float, start_time: float, end_time: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: change an acline fault according to settings in asynchronous system sys_no.

        Args:
            fault_type (int): fault type, 0-three phase fault, 1-three phase disconnection
            fault_dis (float): fault distance, 0-100.
            start_time (float): starting time. The fault happens at this time.
            end_time (float): end time. The fault ends at this time.
            ele_pos (int): acline No. of the fault acline.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_fault(fault_type, fault_dis, start_time, end_time, 0, ele_pos, True, sys_no), \
            f'change fault acline failed, {[fault_type, fault_dis, start_time, end_time, ele_pos, sys_no]}, please check!'

    def set_fault_disturbance_add_transformer(self, fault_type: int, fault_dis: float, start_time: float, end_time: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: add an transformer fault according to settings in asynchronous system sys_no.

        Args:
            fault_type (int): fault type, 0-three phase fault, 1-three phase disconnection
            fault_dis (float): fault distance, 0-100.
            start_time (float): starting time. The fault happens at this time.
            end_time (float): end time. The fault ends at this time.
            ele_pos (int): transformer No. of the fault transformer.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_fault(fault_type, fault_dis, start_time, end_time, 1, ele_pos, False, sys_no), \
            f'add fault transformer failed, {[fault_type, fault_dis, start_time, end_time, ele_pos, sys_no]}, please check!'

    def set_fault_change_transformer(self, fault_type: int, fault_dis: float, start_time: float, end_time: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: change an transformer fault according to settings in asynchronous system sys_no.

        Args:
            fault_type (int): fault type, 0-three phase fault, 1-three phase disconnection
            fault_dis (float): fault distance, 0-100.
            start_time (float): starting time. The fault happens at this time.
            end_time (float): end time. The fault ends at this time.
            ele_pos (int): transformer No. of the fault transformer.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_fault(fault_type, fault_dis, start_time, end_time, 1, ele_pos, True, sys_no), \
            f'change fault transformer failed, {[fault_type, fault_dis, start_time, end_time, ele_pos, sys_no]}, please check!'

    def set_disturbance(self, dis_type: int, dis_time: float, dis_per: float, ele_type: int, ele_pos: int, mod_flg: bool, sys_no=-1):
        """Set disturbance information: basic disturbance set.

        Args:
            dis_type (int): disturbance type, 0-tripping.
            dis_time (float): fault time. The disturbance happens at this time.
            dis_per (float): disturbance percentage, 0-1. dis_per of the generator is subjected to the disturbance.
            ele_type (int): component type, 0-generator, 1-load.
            ele_pos (int): component No. of the fault component.
            mod_flg (bool): flag of modification, False-add new disturbance, True-modify exsiting disturbance.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.

        Returns:
            bool: whether the disturbance is set successfully.
        """        
        return self.__psDLL.set_Fault_Disturbance_Add_Disturbance(dis_type, dis_time, dis_per, ele_type, ele_pos, mod_flg, sys_no)
    
    def set_fault_disturbance_add_generator(self, dis_type: int, dis_time: float, dis_per: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: add an generator fault according to settings in asynchronous system sys_no.

        Args:
            dis_type (int): disturbance type, 0-tripping.
            dis_time (float): fault time. The disturbance happens at this time.
            dis_per (float): disturbance percentage, 0-1. dis_per of the generator is subjected to the disturbance.
            ele_pos (int): generator No. of the fault generator.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_disturbance(dis_type, dis_time, dis_per, 0, ele_pos, False, sys_no), \
            f'add disturbance generator failed, {[dis_type, dis_time, dis_per, ele_pos, sys_no]}, please check!'

    def set_fault_disturbance_change_generator(self, dis_type: int, dis_time: float, dis_per: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: change an generator fault according to settings in asynchronous system sys_no.

        Args:
            dis_type (int): disturbance type, 0-tripping.
            dis_time (float): fault time. The disturbance happens at this time.
            dis_per (float): disturbance percentage, 0-1. dis_per of the generator is subjected to the disturbance.
            ele_pos (int): generator No. of the fault generator.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_disturbance(dis_type, dis_time, dis_per, 0, ele_pos, True, sys_no), \
            f'change disturbance generator failed, {[dis_type, dis_time, dis_per, ele_pos, sys_no]}, please check!'

    def set_fault_disturbance_add_load(self, dis_type: int, dis_time: float, dis_per: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: add an load fault according to settings in asynchronous system sys_no.

        Args:
            dis_type (int): disturbance type, 0-shedding.
            dis_time (float): fault time. The disturbance happens at this time.
            dis_per (float): disturbance percentage, 0-1. dis_per of the load is subjected to the disturbance.
            ele_pos (int): load No. of the fault load.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_disturbance(dis_type, dis_time, dis_per, 1, ele_pos, False, sys_no), \
            f'add disturbance load failed, {[dis_type, dis_time, dis_per, ele_pos, sys_no]}, please check!'

    def set_fault_disturbance_change_load(self, dis_type: int, dis_time: float, dis_per: float, ele_pos: int, sys_no=-1):
        """Set fault and disturbance information: change an load fault according to settings in asynchronous system sys_no.

        Args:
            dis_type (int): disturbance type, 0-shedding.
            dis_time (float): fault time. The disturbance happens at this time.
            dis_per (float): disturbance percentage, 0-1. dis_per of the load is subjected to the disturbance.
            ele_pos (int): load No. of the fault load.
            sys_no (int, optional): asynchronous ac system No. to limit the output range. Defaults to -1, which means the whole network.
        """        
        assert self.set_disturbance(dis_type, dis_time, dis_per, 1, ele_pos, True, sys_no), \
            f'change disturbance load failed, {[dis_type, dis_time, dis_per, ele_pos, sys_no]}, please check!'

    ################################################################################################
    # Supplement
    ################################################################################################
    def _buffer_tests(self):
        """Buffer test.
        """        
        # basic tests
        self.__psDLL.get_LF_V(self.__doubleBuffer, self.__bufferSize, -1)
        x = list(self.__doubleBuffer)
        print(self.__doubleBuffer)
        print(list(self.__doubleBuffer))
        print(x)
        self.__doubleBuffer[0] = 9999.
        x[0] = 1234
        print(self.__doubleBuffer)
        print(list(self.__doubleBuffer))
        print(x)
        self.__psDLL.get_LF_V(self.__doubleBuffer, self.__bufferSize, -1)
        print(self.__doubleBuffer)
        print(list(self.__doubleBuffer))
        print(x)


if __name__ == '__main__':
    # """
    start_time = datetime.datetime.now()
    api = Py_PSOPS(rng=np.random.default_rng(4242))
    # results = np.load('/archive/pythonPS/gym_psops/envs/psops/10-39.npy')
    # index = results[0]
    # load_name = api.get_load_all_bus_name()
    # gen_name = api.get_generator_all_bus_name()
    # idx = [0] * 59
    # inverse_idx = [0] * 59
    # for i in range(len(index)):
    #     if i < 20:
    #         name = 'BUS-' + index[i].strip('load-')
    #         idx[np.where(load_name == name)[0][0]+19] = i
    #         inverse_idx[i] = np.where(load_name == name)[0][0] + 19
    #     elif i < 40:
    #         name = 'BUS-' + index[i].strip('load-')
    #         idx[np.where(load_name == name)[0][0]+39] = i
    #         inverse_idx[i] = np.where(load_name == name)[0][0] + 39
    #     elif i < 49:
    #         name = 'BUS-' + index[i].strip('Gen')
    #         idx[np.where(gen_name == name)[0][0]+10] = i
    #         inverse_idx[i] = np.where(gen_name == name)[0][0] + 10
    #     elif i < 59:
    #         name = 'BUS-' + index[i].strip('Gen')
    #         idx[np.where(gen_name == name)[0][0]] = i
    #         inverse_idx[i] = np.where(gen_name == name)[0][0]
    #     else: raise Exception('too long for cloudpss ieee 39')
    # states, convergences = api.get_pf_sample_all(10000)
    # convergences = np.array(convergences)
    # for i in range(len(states)):
    #     states[i][0][10:] *= 100.0
    #     # print(states[i])
    #     states[i][0] = states[i][0][inverse_idx]
    #     states[i][1] = states[i][1][inverse_idx]
    #     # print(states[i])
    # print(f'converged: {np.where(convergences > 0)[0].shape[0]}. unconverged: {np.where(convergences < 0)[0].shape[0]}.')
    # np.savez('./samples.npz', sample=states, iter=convergences)
    #####################################################################################################################
    # for result in results[1:]:
    #     res = result.copy().astype(float)
    #     res[:49] /= 100.0
    #     res = res[idx]
    #     api.set_pf_initiation(res)
    #     iter_num = api.cal_pf_basic_power_flow_nr()
    #     if iter_num > 0: print(f'converge in {iter_num} iterations\n', api.get_bus_all_lf_result())
    #     else: print('do not converge')
    # api.check_stability()
    # print(api.get_pf_sample_simple_random(load_max=-1, load_min=-1))
    # api.get_pf_sample_simple_random(num=100, check_slack=False, check_voltage=False)
    print(datetime.datetime.now() - start_time)

