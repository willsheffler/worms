repeat run json files seem to need chains in order, but the json io doesnt guarantee this.... fix this

# echo 'Sheet_P321(c3=None, c2=None, **kw)' > Sheet_P321/input/Sheet_P321.flags; cat template.flags >> Sheet_P321/input/Sheet_P321.flags
# echo 'Sheet_P4212(c4=None, c2=None, **kw)' > Sheet_P4212/input/Sheet_P4212.flags; cat template.flags >> Sheet_P4212/input/Sheet_P4212.flags
# echo 'Sheet_P6(c6=None, c2=None, **kw)' > Sheet_P6/input/Sheet_P6.flags; cat template.flags >> Sheet_P6/input/Sheet_P6.flags
# echo 'Sheet_P6_C3_C2(c3=None, c2=None, **kw)' > Sheet_P6_C3_C2/input/Sheet_P6_C3_C2.flags; cat template.flags >> Sheet_P6_C3_C2/input/Sheet_P6_C3_C2.flags
# echo 'Crystal_P213_C3_C3(c3a=None, c3b=None, **kw)' > Crystal_P213_C3_C3/input/Crystal_P213_C3_C3.flags; cat template.flags >> Crystal_P213_C3_C3/input/Crystal_P213_C3_C3.flags
# echo 'Crystal_P4132_C2_C3(c2a=None, c3b=None, **kw)' > Crystal_P4132_C2_C3/input/Crystal_P4132_C2_C3.flags; cat template.flags >> Crystal_P4132_C2_C3/input/Crystal_P4132_C2_C3.flags
# echo 'Crystal_I213_C2_C3(c2a=None, c3b=None, **kw)' > Crystal_I213_C2_C3/input/Crystal_I213_C2_C3.flags; cat template.flags >> Crystal_I213_C2_C3/input/Crystal_I213_C2_C3.flags
# echo 'Crystal_I432_C2_C4(c2a=None, c4b=None, **kw)' > Crystal_I432_C2_C4/input/Crystal_I432_C2_C4.flags; cat template.flags >> Crystal_I432_C2_C4/input/Crystal_I432_C2_C4.flags
# echo 'Crystal_F432_C3_C4(c3a=None, c4b=None, **kw)' > Crystal_F432_C3_C4/input/Crystal_F432_C3_C4.flags; cat template.flags >> Crystal_F432_C3_C4/input/Crystal_F432_C3_C4.flags
# echo 'Crystal_P432_C4_C4(c4a=None, c4b=None, **kw)' > Crystal_P432_C4_C4/input/Crystal_P432_C4_C4.flags; cat template.flags >> Crystal_P432_C4_C4/input/Crystal_P432_C4_C4.flags
# echo 'Sheet_P42_from_ws_0127(c4=None, c2=None, **kw)' > Sheet_P42_from_ws_0127/input/Sheet_P42_from_ws_0127.flags; cat template.flags >> Sheet_P42_from_ws_0127/input/Sheet_P42_from_ws_0127.flags
# echo 'Sheet_P6_C3_C2_from_ws_0202(c3=None, c2=None, **kw)' > Sheet_P6_C3_C2_from_ws_0202/input/Sheet_P6_C3_C2_from_ws_0202.flags; cat template.flags >> Sheet_P6_C3_C2_from_ws_0202/input/Sheet_P6_C3_C2_from_ws_0202.flags
# echo 'Sheet_P4212_from_ws_0127(c4=None, c2=None, **kw)' > Sheet_P4212_from_ws_0127/input/Sheet_P4212_from_ws_0127.flags; cat template.flags >> Sheet_P4212_from_ws_0127/input/Sheet_P4212_from_ws_0127.flags
# echo 'Crystal_F23_T_C2(t=None, c2=None, **kw)' > Crystal_F23_T_C2/input/Crystal_F23_T_C2.flags; cat template.flags >> Crystal_F23_T_C2/input/Crystal_F23_T_C2.flags
# echo 'Crystal_F23_T_C3(t=None, c3=None, **kw)' > Crystal_F23_T_C3/input/Crystal_F23_T_C3.flags; cat template.flags >> Crystal_F23_T_C3/input/Crystal_F23_T_C3.flags
# echo 'Crystal_F23_T_T(t=None, tb=None, **kw)' > Crystal_F23_T_T/input/Crystal_F23_T_T.flags; cat template.flags >> Crystal_F23_T_T/input/Crystal_F23_T_T.flags




grep C4_N   /home/yhsia/helixfuse/2018-07-09_sym/processing/database/HFuse_Cx_database.20180711.txt
grep C4_N   /home/yhsia/helixfuse/cyc_george/processing/database/HFuse_Gcyc_database.20180817.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-103_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-112_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-127_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-131_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-13_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-15_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-34_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-37_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-39_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_zibo_combine/processing/database/HFuse_het_2chain_2arm_database.ZCON-9_2.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh13_3.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh29_3.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_2arm_database.Sh34_3.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh13_3.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh29_3.20180516.txt
grep C4_N   /home/yhsia/helixfuse/asym_sh_combine/processing/database/HFuse_het_3chain_3arm_database.Sh34_3.20180516.txt
grep C4_N   /home/yhsia/helixfuse/2018-09-04_asym_sh_hetc2/combine/processing/database/HFuse_het_2chain_2arm_database.Sh13-5+1-AI_2.20180905.txt
grep C4_N   /home/yhsia/helixfuse/2018-09-04_asym_sh_hetc2/combine/processing/database/HFuse_het_2chain_2arm_database.Sh29-5+1-CI_2.20180905.txt

