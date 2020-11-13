import numpy as np
import pytest

from stixcore.calibration.compression import compress, decompress, NonIntegerCompressionError, \
    CompressionSchemeParameterError, CompressionRangeError


def test_compression():
    comp1 = compress(1, s=0, k=5, m=3)
    comp2 = compress(16106127360, s=0, k=5, m=3)
    assert comp1 == 1
    assert comp2 == 255


def test_compression_nonint_error():
    with pytest.raises(NonIntegerCompressionError):
        _ = compress(np.float64(10.0), s=0, k=5, m=3)


def test_compression_skm_error():
    with pytest.raises(CompressionSchemeParameterError):
        _ = compress(1, s=1, k=5, m=4)


def test_compression_maxvalue_error():
    with pytest.raises(CompressionRangeError) as e:
        _ = compress(256, s=0, k=1, m=7)
    assert str(e.value).startswith('Valid input range exceeded')


def test_decompression_nonint_error():
    with pytest.raises(NonIntegerCompressionError):
        _ = decompress(np.float64(10.0), s=0, k=5, m=3)


def test_decompress():
    decompressed1 = decompress(1, s=0, k=5, m=3)
    decompressed2 = decompress(255, s=0, k=5, m=3)
    assert decompressed1 == 1
    assert decompressed2 == 16106127360


def test_decompress_skm_error():
    with pytest.raises(CompressionSchemeParameterError):
        _ = decompress(1, s=1, k=5, m=4)


def test_decompress_input_error():
    with pytest.raises(CompressionRangeError) as e:
        _ = decompress(256, s=0, k=1, m=7)
    assert str(e.value) == 'Compressed values must be in the range 0 to 255'


def test_decompress_maxvalue_error():
    with pytest.raises(CompressionRangeError) as e:
        _ = decompress(255, s=0, k=6, m=2)
    assert str(e.value) == 'Decompressed value too large to fit into uint64'


@pytest.mark.parametrize('skm', [(0, 0, 8), (0, 1, 7), (0, 2, 6), (0, 3, 5), (0, 4, 4), (0, 5, 3),
                                 (1, 0, 7), (1, 1, 6), (1, 2, 5), (1, 3, 4), (1, 4, 3), (1, 5, 2)])
def test_round_trip(skm):
    s, k, m = skm
    values = np.arange(256)
    decompressed = decompress(values, s=s, k=k, m=m)
    output = compress(decompressed, s=s, k=k, m=m)
    # account for two values of 0 which are possible but as input to decompress is a int only get
    # one
    if s == 1:
        values[values == 128] = 0
    assert np.array_equiv(values, output)


@pytest.mark.parametrize('values', [
    (0,   0,   0),
    (1,   1,   1),
    (2,   2,   2),
    (3,   3,   3),
    (4,   4,   4),
    (5,   5,   5),
    (6,   6,   6),
    (7,   7,   7),
    (8,   8,   8),
    (9,   9,   9),
    (10,  10,  10),
    (11,  11,  11),
    (12,  12,  12),
    (13,  13,  13),
    (14,  14,  14),
    (15,  15,  15),
    (16,  16,  17),
    (17,  18,  19),
    (18,  20,  21),
    (19,  22,  23),
    (20,  24,  25),
    (21,  26,  27),
    (22,  28,  29),
    (23,  30,  31),
    (24,  32,  35),
    (25,  36,  39),
    (26,  40,  43),
    (27,  44,  47),
    (28,  48,  51),
    (29,  52,  55),
    (30,  56,  59),
    (31,  60,  63),
    (32,  64,  71),
    (33,  72,  79),
    (34,  80,  87),
    (35,  88,  95),
    (36,  96,  103),
    (37,  104, 111),
    (38,  112, 119),
    (39,  120, 127),
    (40,  128, 143),
    (41,  144, 159),
    (42,  160, 175),
    (43,  176, 191),
    (44,  192, 207),
    (45,  208, 223),
    (46,  224, 239),
    (47,  240, 255),
    (48,  256, 287),
    (49,  288, 319),
    (50,  320, 351),
    (51,  352, 383),
    (52,  384, 415),
    (53,  416, 447),
    (54,  448, 479),
    (55,  480, 511),
    (56,  512, 575),
    (57,  576, 639),
    (58,  640, 703),
    (59,  704, 767),
    (60,  768, 831),
    (61,  832, 895),
    (62,  896, 959),
    (63,  960, 1023),
    (64,  1024,    1151),
    (65,  1152,    1279),
    (66,  1280,    1407),
    (67,  1408,    1535),
    (68,  1536,    1663),
    (69,  1664,    1791),
    (70,  1792,    1919),
    (71,  1920,    2047),
    (72,  2048,    2303),
    (73,  2304,    2559),
    (74,  2560,    2815),
    (75,  2816,    3071),
    (76,  3072,    3327),
    (77,  3328,    3583),
    (78,  3584,    3839),
    (79,  3840,    4095),
    (80,  4096,    4607),
    (81,  4608,    5119),
    (82,  5120,    5631),
    (83,  5632,    6143),
    (84,  6144,    6655),
    (85,  6656,    7167),
    (86,  7168,    7679),
    (87,  7680,    8191),
    (88,  8192,    9215),
    (89,  9216,    10239),
    (90,  10240,   11263),
    (91,  11264,   12287),
    (92,  12288,   13311),
    (93,  13312,   14335),
    (94,  14336,   15359),
    (95,  15360,   16383),
    (96,  16384,   18431),
    (97,  18432,   20479),
    (98,  20480,   22527),
    (99,  22528,   24575),
    (100, 24576,   26623),
    (101, 26624,   28671),
    (102, 28672,   30719),
    (103, 30720,   32767),
    (104, 32768,   36863),
    (105, 36864,   40959),
    (106, 40960,   45055),
    (107, 45056,   49151),
    (108, 49152,   53247),
    (109, 53248,   57343),
    (110, 57344,   61439),
    (111, 61440,   65535),
    (112, 65536,   73727),
    (113, 73728,   81919),
    (114, 81920,   90111),
    (115, 90112,   98303),
    (116, 98304,   106495),
    (117, 106496,  114687),
    (118, 114688,  122879),
    (119, 122880,  131071),
    (120, 131072,  147455),
    (121, 147456,  163839),
    (122, 163840,  180223),
    (123, 180224,  196607),
    (124, 196608,  212991),
    (125, 212992,  229375),
    (126, 229376,  245759),
    (127, 245760,  262143),
    (128, 262144,  294911),
    (129, 294912,  327679),
    (130, 327680,  360447),
    (131, 360448,  393215),
    (132, 393216,  425983),
    (133, 425984,  458751),
    (134, 458752,  491519),
    (135, 491520,  524287),
    (136, 524288,  589823),
    (137, 589824,  655359),
    (138, 655360,  720895),
    (139, 720896,  786431),
    (140, 786432,  851967),
    (141, 851968,  917503),
    (142, 917504,  983039),
    (143, 983040,  1048575),
    (144, 1048576, 1179647),
    (145, 1179648, 1310719),
    (146, 1310720, 1441791),
    (147, 1441792, 1572863),
    (148, 1572864, 1703935),
    (149, 1703936, 1835007),
    (150, 1835008, 1966079),
    (151, 1966080, 2097151),
    (152, 2097152, 2359295),
    (153, 2359296, 2621439),
    (154, 2621440, 2883583),
    (155, 2883584, 3145727),
    (156, 3145728, 3407871),
    (157, 3407872, 3670015),
    (158, 3670016, 3932159),
    (159, 3932160, 4194303),
    (160, 4194304, 4718591),
    (161, 4718592, 5242879),
    (162, 5242880, 5767167),
    (163, 5767168, 6291455),
    (164, 6291456, 6815743),
    (165, 6815744, 7340031),
    (166, 7340032, 7864319),
    (167, 7864320, 8388607),
    (168, 8388608, 9437183),
    (169, 9437184, 10485759),
    (170, 10485760,    11534335),
    (171, 11534336,    12582911),
    (172, 12582912,    13631487),
    (173, 13631488,    14680063),
    (174, 14680064,    15728639),
    (175, 15728640,    16777215),
    (176, 16777216,    18874367),
    (177, 18874368,    20971519),
    (178, 20971520,    23068671),
    (179, 23068672,    25165823),
    (180, 25165824,    27262975),
    (181, 27262976,    29360127),
    (182, 29360128,    31457279),
    (183, 31457280,    33554431),
    (184, 33554432,    37748735),
    (185, 37748736,    41943039),
    (186, 41943040,    46137343),
    (187, 46137344,    50331647),
    (188, 50331648,    54525951),
    (189, 54525952,    58720255),
    (190, 58720256,    62914559),
    (191, 62914560,    67108863),
    (192, 67108864,    75497471),
    (193, 75497472,    83886079),
    (194, 83886080,    92274687),
    (195, 92274688,    100663295),
    (196, 100663296,   109051903),
    (197, 109051904,   117440511),
    (198, 117440512,   125829119),
    (199, 125829120,   134217727),
    (200, 134217728,   150994943),
    (201, 150994944,   167772159),
    (202, 167772160,   184549375),
    (203, 184549376,   201326591),
    (204, 201326592,   218103807),
    (205, 218103808,   234881023),
    (206, 234881024,   251658239),
    (207, 251658240,   268435455),
    (208, 268435456,   301989887),
    (209, 301989888,   335544319),
    (210, 335544320,   369098751),
    (211, 369098752,   402653183),
    (212, 402653184,   436207615),
    (213, 436207616,   469762047),
    (214, 469762048,   503316479),
    (215, 503316480,   536870911),
    (216, 536870912,   603979775),
    (217, 603979776,   671088639),
    (218, 671088640,   738197503),
    (219, 738197504,   805306367),
    (220, 805306368,   872415231),
    (221, 872415232,   939524095),
    (222, 939524096,   1006632959),
    (223, 1006632960,  1073741823),
    (224, 1073741824,  1207959551),
    (225, 1207959552,  1342177279),
    (226, 1342177280,  1476395007),
    (227, 1476395008,  1610612735),
    (228, 1610612736,  1744830463),
    (229, 1744830464,  1879048191),
    (230, 1879048192,  2013265919),
    (231, 2013265920,  2147483647),
    (232, 2147483648,  2415919103),
    (233, 2415919104,  2684354559),
    (234, 2684354560,  2952790015),
    (235, 2952790016,  3221225471),
    (236, 3221225472,  3489660927),
    (237, 3489660928,  3758096383),
    (238, 3758096384,  4026531839),
    (239, 4026531840,  4294967295),
    (240, 4294967296,  4831838207),
    (241, 4831838208,  5368709119),
    (242, 5368709120,  5905580031),
    (243, 5905580032,  6442450943),
    (244, 6442450944,  6979321855),
    (245, 6979321856,  7516192767),
    (246, 7516192768,  8053063679),
    (247, 8053063680,  8589934591),
    (248, 8589934592,  9663676415),
    (249, 9663676416,  10737418239),
    (250, 10737418240, 11811160063),
    (251, 11811160064, 12884901887),
    (252, 12884901888, 13958643711),
    (253, 13958643712, 15032385535),
    (254, 15032385536, 16106127359),
    (255, 16106127360, 16106127360),
])
def test_compress_053(values):
    compressed, start, end = values
    res = compress(np.linspace(start, end, 10, dtype=np.uint64), s=0, k=5, m=3)
    assert np.all(compressed == res)


@pytest.mark.parametrize('values', [
    (0,   0.00),
    (1,   0.00),
    (2,   0.00),
    (3,   0.00),
    (4,   0.00),
    (5,   0.00),
    (6,   0.00),
    (7,   0.00),
    (8,   0.00),
    (9,   0.00),
    (10,  0.00),
    (11,  0.00),
    (12,  0.00),
    (13,  0.00),
    (14,  0.00),
    (15,  0.00),
    (16,  0.50),
    (17,  0.50),
    (18,  0.50),
    (19,  0.50),
    (20,  0.50),
    (21,  0.50),
    (22,  0.50),
    (23,  0.50),
    (24,  1.50),
    (25,  1.50),
    (26,  1.50),
    (27,  1.50),
    (28,  1.50),
    (29,  1.50),
    (30,  1.50),
    (31,  1.50),
    (32,  5.50),
    (33,  5.50),
    (34,  5.50),
    (35,  5.50),
    (36,  5.50),
    (37,  5.50),
    (38,  5.50),
    (39,  5.50),
    (40,  21.50),
    (41,  21.50),
    (42,  21.50),
    (43,  21.50),
    (44,  21.50),
    (45,  21.50),
    (46,  21.50),
    (47,  21.50),
    (48,  85.50),
    (49,  85.50),
    (50,  85.50),
    (51,  85.50),
    (52,  85.50),
    (53,  85.50),
    (54,  85.50),
    (55,  85.50),
    (56,  341.50),
    (57,  341.50),
    (58,  341.50),
    (59,  341.50),
    (60,  341.50),
    (61,  341.50),
    (62,  341.50),
    (63,  341.50),
    (64,  1365.50),
    (65,  1365.50),
    (66,  1365.50),
    (67,  1365.50),
    (68,  1365.50),
    (69,  1365.50),
    (70,  1365.50),
    (71,  1365.50),
    (72,  5461.50),
    (73,  5461.50),
    (74,  5461.50),
    (75,  5461.50),
    (76,  5461.50),
    (77,  5461.50),
    (78,  5461.50),
    (79,  5461.50),
    (80,  21845.50),
    (81,  21845.50),
    (82,  21845.50),
    (83,  21845.50),
    (84,  21845.50),
    (85,  21845.50),
    (86,  21845.50),
    (87,  21845.50),
    (88,  87381.50),
    (89,  87381.50),
    (90,  87381.50),
    (91,  87381.50),
    (92,  87381.50),
    (93,  87381.50),
    (94,  87381.50),
    (95,  87381.50),
    (96,  349525.50),
    (97,  349525.50),
    (98,  349525.50),
    (99,  349525.50),
    (100, 349525.50),
    (101, 349525.50),
    (102, 349525.50),
    (103, 349525.50),
    (104, 1398101.50),
    (105, 1398101.50),
    (106, 1398101.50),
    (107, 1398101.50),
    (108, 1398101.50),
    (109, 1398101.50),
    (110, 1398101.50),
    (111, 1398101.50),
    (112, 5592405.50),
    (113, 5592405.50),
    (114, 5592405.50),
    (115, 5592405.50),
    (116, 5592405.50),
    (117, 5592405.50),
    (118, 5592405.50),
    (119, 5592405.50),
    (120, 22369621.50),
    (121, 22369621.50),
    (122, 22369621.50),
    (123, 22369621.50),
    (124, 22369621.50),
    (125, 22369621.50),
    (126, 22369621.50),
    (127, 22369621.50),
    (128, 89478485.50),
    (129, 89478485.50),
    (130, 89478485.50),
    (131, 89478485.50),
    (132, 89478485.50),
    (133, 89478485.50),
    (134, 89478485.50),
    (135, 89478485.50),
    (136, 357913941.50),
    (137, 357913941.50),
    (138, 357913941.50),
    (139, 357913941.50),
    (140, 357913941.50),
    (141, 357913941.50),
    (142, 357913941.50),
    (143, 357913941.50),
    (144, 1431655765.50),
    (145, 1431655765.50),
    (146, 1431655765.50),
    (147, 1431655765.50),
    (148, 1431655765.50),
    (149, 1431655765.50),
    (150, 1431655765.50),
    (151, 1431655765.50),
    (152, 5726623061.50),
    (153, 5726623061.50),
    (154, 5726623061.50),
    (155, 5726623061.50),
    (156, 5726623061.50),
    (157, 5726623061.50),
    (158, 5726623061.50),
    (159, 5726623061.50),
    (160, 22906492245.50),
    (161, 22906492245.50),
    (162, 22906492245.50),
    (163, 22906492245.50),
    (164, 22906492245.50),
    (165, 22906492245.50),
    (166, 22906492245.50),
    (167, 22906492245.50),
    (168, 91625968981.50),
    (169, 91625968981.50),
    (170, 91625968981.50),
    (171, 91625968981.50),
    (172, 91625968981.50),
    (173, 91625968981.50),
    (174, 91625968981.50),
    (175, 91625968981.50),
    (176, 366503875925.50),
    (177, 366503875925.50),
    (178, 366503875925.50),
    (179, 366503875925.50),
    (180, 366503875925.50),
    (181, 366503875925.50),
    (182, 366503875925.50),
    (183, 366503875925.50),
    (184, 1466015503701.50),
    (185, 1466015503701.50),
    (186, 1466015503701.50),
    (187, 1466015503701.50),
    (188, 1466015503701.50),
    (189, 1466015503701.50),
    (190, 1466015503701.50),
    (191, 1466015503701.50),
    (192, 5864062014805.50),
    (193, 5864062014805.50),
    (194, 5864062014805.50),
    (195, 5864062014805.50),
    (196, 5864062014805.50),
    (197, 5864062014805.50),
    (198, 5864062014805.50),
    (199, 5864062014805.50),
    (200, 23456248059221.50),
    (201, 23456248059221.50),
    (202, 23456248059221.50),
    (203, 23456248059221.50),
    (204, 23456248059221.50),
    (205, 23456248059221.50),
    (206, 23456248059221.50),
    (207, 23456248059221.50),
    (208, 93824992236885.50),
    (209, 93824992236885.50),
    (210, 93824992236885.50),
    (211, 93824992236885.50),
    (212, 93824992236885.50),
    (213, 93824992236885.50),
    (214, 93824992236885.50),
    (215, 93824992236885.50),
    (216, 375299968947541.50),
    (217, 375299968947541.50),
    (218, 375299968947541.50),
    (219, 375299968947541.50),
    (220, 375299968947541.50),
    (221, 375299968947541.50),
    (222, 375299968947541.50),
    (223, 375299968947541.50),
    (224, 1501199875790165.25),
    (225, 1501199875790165.25),
    (226, 1501199875790165.25),
    (227, 1501199875790165.25),
    (228, 1501199875790165.25),
    (229, 1501199875790165.25),
    (230, 1501199875790165.25),
    (231, 1501199875790165.25),
    (232, 6004799503160661.00),
    (233, 6004799503160661.00),
    (234, 6004799503160661.00),
    (235, 6004799503160661.00),
    (236, 6004799503160661.00),
    (237, 6004799503160661.00),
    (238, 6004799503160661.00),
    (239, 6004799503160661.00),
    (240, 24019198012642644.00),
    (241, 24019198012642644.00),
    (242, 24019198012642644.00),
    (243, 24019198012642644.00),
    (244, 24019198012642644.00),
    (245, 24019198012642644.00),
    (246, 24019198012642644.00),
    (247, 24019198012642644.00),
    (248, 96076792050570576.00),
    (249, 96076792050570576.00),
    (250, 96076792050570576.00),
    (251, 96076792050570576.00),
    (252, 96076792050570576.00),
    (253, 96076792050570576.00),
    (254, 96076792050570576.00),
    (255, 259407338536540569600.00),
])
def test_variance_053(values):
    compressed, variance = values
    _, res = decompress(compressed, s=0, k=5, m=3, return_variance=True)
    assert variance == res


@pytest.mark.parametrize('values', [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
    (17, 18),
    (18, 20),
    (19, 22),
    (20, 24),
    (21, 26),
    (22, 28),
    (23, 30),
    (24, 33),
    (25, 37),
    (26, 41),
    (27, 45),
    (28, 49),
    (29, 53),
    (30, 57),
    (31, 61),
    (32, 67),
    (33, 75),
    (34, 83),
    (35, 91),
    (36, 99),
    (37, 107),
    (38, 115),
    (39, 123),
    (40, 135),
    (41, 151),
    (42, 167),
    (43, 183),
    (44, 199),
    (45, 215),
    (46, 231),
    (47, 247),
    (48, 271),
    (49, 303),
    (50, 335),
    (51, 367),
    (52, 399),
    (53, 431),
    (54, 463),
    (55, 495),
    (56, 543),
    (57, 607),
    (58, 671),
    (59, 735),
    (60, 799),
    (61, 863),
    (62, 927),
    (63, 991),
    (64, 1087),
    (65, 1215),
    (66, 1343),
    (67, 1471),
    (68, 1599),
    (69, 1727),
    (70, 1855),
    (71, 1983),
    (72, 2175),
    (73, 2431),
    (74, 2687),
    (75, 2943),
    (76, 3199),
    (77, 3455),
    (78, 3711),
    (79, 3967),
    (80, 4351),
    (81, 4863),
    (82, 5375),
    (83, 5887),
    (84, 6399),
    (85, 6911),
    (86, 7423),
    (87, 7935),
    (88, 8703),
    (89, 9727),
    (90, 10751),
    (91, 11775),
    (92, 12799),
    (93, 13823),
    (94, 14847),
    (95, 15871),
    (96, 17407),
    (97, 19455),
    (98, 21503),
    (99, 23551),
    (100, 25599),
    (101, 27647),
    (102, 29695),
    (103, 31743),
    (104, 34815),
    (105, 38911),
    (106, 43007),
    (107, 47103),
    (108, 51199),
    (109, 55295),
    (110, 59391),
    (111, 63487),
    (112, 69631),
    (113, 77823),
    (114, 86015),
    (115, 94207),
    (116, 102399),
    (117, 110591),
    (118, 118783),
    (119, 126975),
    (120, 139263),
    (121, 155647),
    (122, 172031),
    (123, 188415),
    (124, 204799),
    (125, 221183),
    (126, 237567),
    (127, 253951),
    (128, 278527),
    (129, 311295),
    (130, 344063),
    (131, 376831),
    (132, 409599),
    (133, 442367),
    (134, 475135),
    (135, 507903),
    (136, 557055),
    (137, 622591),
    (138, 688127),
    (139, 753663),
    (140, 819199),
    (141, 884735),
    (142, 950271),
    (143, 1015807),
    (144, 1114111),
    (145, 1245183),
    (146, 1376255),
    (147, 1507327),
    (148, 1638399),
    (149, 1769471),
    (150, 1900543),
    (151, 2031615),
    (152, 2228223),
    (153, 2490367),
    (154, 2752511),
    (155, 3014655),
    (156, 3276799),
    (157, 3538943),
    (158, 3801087),
    (159, 4063231),
    (160, 4456447),
    (161, 4980735),
    (162, 5505023),
    (163, 6029311),
    (164, 6553599),
    (165, 7077887),
    (166, 7602175),
    (167, 8126463),
    (168, 8912895),
    (169, 9961471),
    (170, 11010047),
    (171, 12058623),
    (172, 13107199),
    (173, 14155775),
    (174, 15204351),
    (175, 16252927),
    (176, 17825791),
    (177, 19922943),
    (178, 22020095),
    (179, 24117247),
    (180, 26214399),
    (181, 28311551),
    (182, 30408703),
    (183, 32505855),
    (184, 35651583),
    (185, 39845887),
    (186, 44040191),
    (187, 48234495),
    (188, 52428799),
    (189, 56623103),
    (190, 60817407),
    (191, 65011711),
    (192, 71303167),
    (193, 79691775),
    (194, 88080383),
    (195, 96468991),
    (196, 104857599),
    (197, 113246207),
    (198, 121634815),
    (199, 130023423),
    (200, 142606335),
    (201, 159383551),
    (202, 176160767),
    (203, 192937983),
    (204, 209715199),
    (205, 226492415),
    (206, 243269631),
    (207, 260046847),
    (208, 285212671),
    (209, 318767103),
    (210, 352321535),
    (211, 385875967),
    (212, 419430399),
    (213, 452984831),
    (214, 486539263),
    (215, 520093695),
    (216, 570425343),
    (217, 637534207),
    (218, 704643071),
    (219, 771751935),
    (220, 838860799),
    (221, 905969663),
    (222, 973078527),
    (223, 1040187391),
    (224, 1140850687),
    (225, 1275068415),
    (226, 1409286143),
    (227, 1543503871),
    (228, 1677721599),
    (229, 1811939327),
    (230, 1946157055),
    (231, 2080374783),
    (232, 2281701375),
    (233, 2550136831),
    (234, 2818572287),
    (235, 3087007743),
    (236, 3355443199),
    (237, 3623878655),
    (238, 3892314111),
    (239, 4160749567),
    (240, 4563402751),
    (241, 5100273663),
    (242, 5637144575),
    (243, 6174015487),
    (244, 6710886399),
    (245, 7247757311),
    (246, 7784628223),
    (247, 8321499135),
    (248, 9126805503),
    (249, 10200547327),
    (250, 11274289151),
    (251, 12348030975),
    (252, 13421772799),
    (253, 14495514623),
    (254, 15569256447),
    (255, 16106127360),
])
def test_decompress_053(values):
    compressed, decompressed = values
    res = decompress(compressed, s=0, k=5, m=3)
    assert res == decompressed


@pytest.mark.parametrize('values', [
    (0, 0,   0),
    (1, 1,   1),
    (2, 2,   2),
    (3, 3,   3),
    (4, 4,   4),
    (5, 5,   5),
    (6, 6,   6),
    (7, 7,   7),
    (8, 8,   8),
    (9, 9,   9),
    (10, 10,  10),
    (11, 11,  11),
    (12, 12,  12),
    (13, 13,  13),
    (14, 14,  14),
    (15, 15,  15),
    (16, 16,  17),
    (17, 18,  19),
    (18, 20,  21),
    (19, 22,  23),
    (20, 24,  25),
    (21, 26,  27),
    (22, 28,  29),
    (23, 30,  31),
    (24, 32,  35),
    (25, 36,  39),
    (26, 40,  43),
    (27, 44,  47),
    (28, 48,  51),
    (29, 52,  55),
    (30, 56,  59),
    (31, 60,  63),
    (32, 64,  71),
    (33, 72,  79),
    (34, 80,  87),
    (35, 88,  95),
    (36, 96,  103),
    (37, 104, 111),
    (38, 112, 119),
    (39, 120, 127),
    (40, 128, 143),
    (41, 144, 159),
    (42, 160, 175),
    (43, 176, 191),
    (44, 192, 207),
    (45, 208, 223),
    (46, 224, 239),
    (47, 240, 255),
    (48, 256, 287),
    (49, 288, 319),
    (50, 320, 351),
    (51, 352, 383),
    (52, 384, 415),
    (53, 416, 447),
    (54, 448, 479),
    (55, 480, 511),
    (56, 512, 575),
    (57, 576, 639),
    (58, 640, 703),
    (59, 704, 767),
    (60, 768, 831),
    (61, 832, 895),
    (62, 896, 959),
    (63, 960, 1023),
    (64, 1024,    1151),
    (65, 1152,    1279),
    (66, 1280,    1407),
    (67, 1408,    1535),
    (68, 1536,    1663),
    (69, 1664,    1791),
    (70, 1792,    1919),
    (71, 1920,    2047),
    (72, 2048,    2303),
    (73, 2304,    2559),
    (74, 2560,    2815),
    (75, 2816,    3071),
    (76, 3072,    3327),
    (77, 3328,    3583),
    (78, 3584,    3839),
    (79, 3840,    4095),
    (80, 4096,    4607),
    (81, 4608,    5119),
    (82, 5120,    5631),
    (83, 5632,    6143),
    (84, 6144,    6655),
    (85, 6656,    7167),
    (86, 7168,    7679),
    (87, 7680,    8191),
    (88, 8192,    9215),
    (89, 9216,    10239),
    (90, 10240,   11263),
    (91, 11264,   12287),
    (92, 12288,   13311),
    (93, 13312,   14335),
    (94, 14336,   15359),
    (95, 15360,   16383),
    (96, 16384,   18431),
    (97, 18432,   20479),
    (98, 20480,   22527),
    (99, 22528,   24575),
    (100, 24576,   26623),
    (101, 26624,   28671),
    (102, 28672,   30719),
    (103, 30720,   32767),
    (104, 32768,   36863),
    (105, 36864,   40959),
    (106, 40960,   45055),
    (107, 45056,   49151),
    (108, 49152,   53247),
    (109, 53248,   57343),
    (110, 57344,   61439),
    (111, 61440,   65535),
    (112, 65536,   73727),
    (113, 73728,   81919),
    (114, 81920,   90111),
    (115, 90112,   98303),
    (116, 98304,   106495),
    (117, 106496,  114687),
    (118, 114688,  122879),
    (119, 122880,  131071),
    (120, 131072,  147455),
    (121, 147456,  163839),
    (122, 163840,  180223),
    (123, 180224,  196607),
    (124, 196608,  212991),
    (125, 212992,  229375),
    (126, 229376,  245759),
    (127, 245760, 245760),
    (0, -0, -0),
    (129, -1, -1),
    (130, -2, -2),
    (131, -3, -3),
    (132, -4, -4),
    (133, -5, -5),
    (134, -6, -6),
    (135, -7, -7),
    (136, -8, -8),
    (137, -9, -9),
    (138, -10, -10),
    (139, -11, -11),
    (140, -12, -12),
    (141, -13, -13),
    (142, -14, -14),
    (143, -15, -15),
    (144, -16, -17),
    (145, -18, -19),
    (146, -20, -21),
    (147, -22, -23),
    (148, -24, -25),
    (149, -26, -27),
    (150, -28, -29),
    (151, -30, -31),
    (152, -32, -35),
    (153, -36, -39),
    (154, -40, -43),
    (155, -44, -47),
    (156, -48, -51),
    (157, -52, -55),
    (158, -56, -59),
    (159, -60, -63),
    (160, -64, -71),
    (161, -72, -79),
    (162, -80, -87),
    (163, -88, -95),
    (164, -96, -103),
    (165, -104, -111),
    (166, -112, -119),
    (167, -120, -127),
    (168, -128, -143),
    (169, -144, -159),
    (170, -160, -175),
    (171, -176, -191),
    (172, -192, -207),
    (173, -208, -223),
    (174, -224, -239),
    (175, -240, -255),
    (176, -256, -287),
    (177, -288, -319),
    (178, -320, -351),
    (179, -352, -383),
    (180, -384, -415),
    (181, -416, -447),
    (182, -448, -479),
    (183, -480, -511),
    (184, -512, -575),
    (185, -576, -639),
    (186, -640, -703),
    (187, -704, -767),
    (188, -768, -831),
    (189, -832, -895),
    (190, -896, -959),
    (191, -960, -1023),
    (192, -1024, -1151),
    (193, -1152, -1279),
    (194, -1280, -1407),
    (195, -1408, -1535),
    (196, -1536, -1663),
    (197, -1664, -1791),
    (198, -1792, -1919),
    (199, -1920, -2047),
    (200, -2048, -2303),
    (201, -2304, -2559),
    (202, -2560, -2815),
    (203, -2816, -3071),
    (204, -3072, -3327),
    (205, -3328, -3583),
    (206, -3584, -3839),
    (207, -3840, -4095),
    (208, -4096, -4607),
    (209, -4608, -5119),
    (210, -5120, -5631),
    (211, -5632, -6143),
    (212, -6144, -6655),
    (213, -6656, -7167),
    (214, -7168, -7679),
    (215, -7680, -8191),
    (216, -8192, -9215),
    (217, -9216, -10239),
    (218, -10240, -11263),
    (219, -11264, -12287),
    (220, -12288, -13311),
    (221, -13312, -14335),
    (222, -14336, -15359),
    (223, -15360, -16383),
    (224, -16384, -18431),
    (225, -18432, -20479),
    (226, -20480, -22527),
    (227, -22528, -24575),
    (228, -24576, -26623),
    (229, -26624, -28671),
    (230, -28672, -30719),
    (231, -30720, -32767),
    (232, -32768, -36863),
    (233, -36864, -40959),
    (234, -40960, -45055),
    (235, -45056, -49151),
    (236, -49152, -53247),
    (237, -53248, -57343),
    (238, -57344, -61439),
    (239, -61440, -65535),
    (240, -65536, -73727),
    (241, -73728, -81919),
    (242, -81920, -90111),
    (243, -90112, -98303),
    (244, -98304, -106495),
    (245, -106496, -114687),
    (246, -114688, -122879),
    (247, -122880, -131071),
    (248, -131072, -147455),
    (249, -147456, -163839),
    (250, -163840, -180223),
    (251, -180224, -196607),
    (252, -196608, -212991),
    (253, -212992, -229375),
    (254, -229376, -245759),
    (255, -245760, -245760),
])
def test_compress_143(values):
    compressed, start, end = values
    res = compress(np.linspace(start, end, 10, dtype=np.int64), s=1, k=4, m=3)
    assert np.all(compressed == res)


@pytest.mark.parametrize('values', [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
    (10, 10),
    (11, 11),
    (12, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
    (17, 18),
    (18, 20),
    (19, 22),
    (20, 24),
    (21, 26),
    (22, 28),
    (23, 30),
    (24, 33),
    (25, 37),
    (26, 41),
    (27, 45),
    (28, 49),
    (29, 53),
    (30, 57),
    (31, 61),
    (32, 67),
    (33, 75),
    (34, 83),
    (35, 91),
    (36, 99),
    (37, 107),
    (38, 115),
    (39, 123),
    (40, 135),
    (41, 151),
    (42, 167),
    (43, 183),
    (44, 199),
    (45, 215),
    (46, 231),
    (47, 247),
    (48, 271),
    (49, 303),
    (50, 335),
    (51, 367),
    (52, 399),
    (53, 431),
    (54, 463),
    (55, 495),
    (56, 543),
    (57, 607),
    (58, 671),
    (59, 735),
    (60, 799),
    (61, 863),
    (62, 927),
    (63, 991),
    (64, 1087),
    (65, 1215),
    (66, 1343),
    (67, 1471),
    (68, 1599),
    (69, 1727),
    (70, 1855),
    (71, 1983),
    (72, 2175),
    (73, 2431),
    (74, 2687),
    (75, 2943),
    (76, 3199),
    (77, 3455),
    (78, 3711),
    (79, 3967),
    (80, 4351),
    (81, 4863),
    (82, 5375),
    (83, 5887),
    (84, 6399),
    (85, 6911),
    (86, 7423),
    (87, 7935),
    (88, 8703),
    (89, 9727),
    (90, 10751),
    (91, 11775),
    (92, 12799),
    (93, 13823),
    (94, 14847),
    (95, 15871),
    (96, 17407),
    (97, 19455),
    (98, 21503),
    (99, 23551),
    (100, 25599),
    (101, 27647),
    (102, 29695),
    (103, 31743),
    (104, 34815),
    (105, 38911),
    (106, 43007),
    (107, 47103),
    (108, 51199),
    (109, 55295),
    (110, 59391),
    (111, 63487),
    (112, 69631),
    (113, 77823),
    (114, 86015),
    (115, 94207),
    (116, 102399),
    (117, 110591),
    (118, 118783),
    (119, 126975),
    (120, 139263),
    (121, 155647),
    (122, 172031),
    (123, 188415),
    (124, 204799),
    (125, 221183),
    (126, 237567),
    (127, 245760),
    (128, 0),
    (129, -1),
    (130, -2),
    (131, -3),
    (132, -4),
    (133, -5),
    (134, -6),
    (135, -7),
    (136, -8),
    (137, -9),
    (138, -10),
    (139, -11),
    (140, -12),
    (141, -13),
    (142, -14),
    (143, -15),
    (144, -16),
    (145, -18),
    (146, -20),
    (147, -22),
    (148, -24),
    (149, -26),
    (150, -28),
    (151, -30),
    (152, -33),
    (153, -37),
    (154, -41),
    (155, -45),
    (156, -49),
    (157, -53),
    (158, -57),
    (159, -61),
    (160, -67),
    (161, -75),
    (162, -83),
    (163, -91),
    (164, -99),
    (165, -107),
    (166, -115),
    (167, -123),
    (168, -135),
    (169, -151),
    (170, -167),
    (171, -183),
    (172, -199),
    (173, -215),
    (174, -231),
    (175, -247),
    (176, -271),
    (177, -303),
    (178, -335),
    (179, -367),
    (180, -399),
    (181, -431),
    (182, -463),
    (183, -495),
    (184, -543),
    (185, -607),
    (186, -671),
    (187, -735),
    (188, -799),
    (189, -863),
    (190, -927),
    (191, -991),
    (192, -1087),
    (193, -1215),
    (194, -1343),
    (195, -1471),
    (196, -1599),
    (197, -1727),
    (198, -1855),
    (199, -1983),
    (200, -2175),
    (201, -2431),
    (202, -2687),
    (203, -2943),
    (204, -3199),
    (205, -3455),
    (206, -3711),
    (207, -3967),
    (208, -4351),
    (209, -4863),
    (210, -5375),
    (211, -5887),
    (212, -6399),
    (213, -6911),
    (214, -7423),
    (215, -7935),
    (216, -8703),
    (217, -9727),
    (218, -10751),
    (219, -11775),
    (220, -12799),
    (221, -13823),
    (222, -14847),
    (223, -15871),
    (224, -17407),
    (225, -19455),
    (226, -21503),
    (227, -23551),
    (228, -25599),
    (229, -27647),
    (230, -29695),
    (231, -31743),
    (232, -34815),
    (233, -38911),
    (234, -43007),
    (235, -47103),
    (236, -51199),
    (237, -55295),
    (238, -59391),
    (239, -63487),
    (240, -69631),
    (241, -77823),
    (242, -86015),
    (243, -94207),
    (244, -102399),
    (245, -110591),
    (246, -118783),
    (247, -126975),
    (248, -139263),
    (249, -155647),
    (250, -172031),
    (251, -188415),
    (252, -204799),
    (253, -221183),
    (254, -237567),
    (255, -245760)
])
def test_decompress_143(values):
    compressed, decompressed = values
    res = decompress(compressed, s=1, k=4, m=3)
    assert res == decompressed


@pytest.mark.parametrize('values', [
    (0,   0.00),
    (1,   0.00),
    (2,   0.00),
    (3,   0.00),
    (4,   0.00),
    (5,   0.00),
    (6,   0.00),
    (7,   0.00),
    (8,   0.00),
    (9,   0.00),
    (10,  0.00),
    (11,  0.00),
    (12,  0.00),
    (13,  0.00),
    (14,  0.00),
    (15,  0.00),
    (16,  0.50),
    (17,  0.50),
    (18,  0.50),
    (19,  0.50),
    (20,  0.50),
    (21,  0.50),
    (22,  0.50),
    (23,  0.50),
    (24,  1.50),
    (25,  1.50),
    (26,  1.50),
    (27,  1.50),
    (28,  1.50),
    (29,  1.50),
    (30,  1.50),
    (31,  1.50),
    (32,  5.50),
    (33,  5.50),
    (34,  5.50),
    (35,  5.50),
    (36,  5.50),
    (37,  5.50),
    (38,  5.50),
    (39,  5.50),
    (40,  21.50),
    (41,  21.50),
    (42,  21.50),
    (43,  21.50),
    (44,  21.50),
    (45,  21.50),
    (46,  21.50),
    (47,  21.50),
    (48,  85.50),
    (49,  85.50),
    (50,  85.50),
    (51,  85.50),
    (52,  85.50),
    (53,  85.50),
    (54,  85.50),
    (55,  85.50),
    (56,  341.50),
    (57,  341.50),
    (58,  341.50),
    (59,  341.50),
    (60,  341.50),
    (61,  341.50),
    (62,  341.50),
    (63,  341.50),
    (64,  1365.50),
    (65,  1365.50),
    (66,  1365.50),
    (67,  1365.50),
    (68,  1365.50),
    (69,  1365.50),
    (70,  1365.50),
    (71,  1365.50),
    (72,  5461.50),
    (73,  5461.50),
    (74,  5461.50),
    (75,  5461.50),
    (76,  5461.50),
    (77,  5461.50),
    (78,  5461.50),
    (79,  5461.50),
    (80,  21845.50),
    (81,  21845.50),
    (82,  21845.50),
    (83,  21845.50),
    (84,  21845.50),
    (85,  21845.50),
    (86,  21845.50),
    (87,  21845.50),
    (88,  87381.50),
    (89,  87381.50),
    (90,  87381.50),
    (91,  87381.50),
    (92,  87381.50),
    (93,  87381.50),
    (94,  87381.50),
    (95,  87381.50),
    (96,  349525.50),
    (97,  349525.50),
    (98,  349525.50),
    (99,  349525.50),
    (100, 349525.50),
    (101, 349525.50),
    (102, 349525.50),
    (103, 349525.50),
    (104, 1398101.50),
    (105, 1398101.50),
    (106, 1398101.50),
    (107, 1398101.50),
    (108, 1398101.50),
    (109, 1398101.50),
    (110, 1398101.50),
    (111, 1398101.50),
    (112, 5592405.50),
    (113, 5592405.50),
    (114, 5592405.50),
    (115, 5592405.50),
    (116, 5592405.50),
    (117, 5592405.50),
    (118, 5592405.50),
    (119, 5592405.50),
    (120, 22369621.50),
    (121, 22369621.50),
    (122, 22369621.50),
    (123, 22369621.50),
    (124, 22369621.50),
    (125, 22369621.50),
    (126, 22369621.50),
    (127, 60397977600.00),
    (128, 0.00),
    (129, 0.00),
    (130, 0.00),
    (131, 0.00),
    (132, 0.00),
    (133, 0.00),
    (134, 0.00),
    (135, 0.00),
    (136, 0.00),
    (137, 0.00),
    (138, 0.00),
    (139, 0.00),
    (140, 0.00),
    (141, 0.00),
    (142, 0.00),
    (143, 0.00),
    (144, 0.50),
    (145, 0.50),
    (146, 0.50),
    (147, 0.50),
    (148, 0.50),
    (149, 0.50),
    (150, 0.50),
    (151, 0.50),
    (152, 1.50),
    (153, 1.50),
    (154, 1.50),
    (155, 1.50),
    (156, 1.50),
    (157, 1.50),
    (158, 1.50),
    (159, 1.50),
    (160, 5.50),
    (161, 5.50),
    (162, 5.50),
    (163, 5.50),
    (164, 5.50),
    (165, 5.50),
    (166, 5.50),
    (167, 5.50),
    (168, 21.50),
    (169, 21.50),
    (170, 21.50),
    (171, 21.50),
    (172, 21.50),
    (173, 21.50),
    (174, 21.50),
    (175, 21.50),
    (176, 85.50),
    (177, 85.50),
    (178, 85.50),
    (179, 85.50),
    (180, 85.50),
    (181, 85.50),
    (182, 85.50),
    (183, 85.50),
    (184, 341.50),
    (185, 341.50),
    (186, 341.50),
    (187, 341.50),
    (188, 341.50),
    (189, 341.50),
    (190, 341.50),
    (191, 341.50),
    (192, 1365.50),
    (193, 1365.50),
    (194, 1365.50),
    (195, 1365.50),
    (196, 1365.50),
    (197, 1365.50),
    (198, 1365.50),
    (199, 1365.50),
    (200, 5461.50),
    (201, 5461.50),
    (202, 5461.50),
    (203, 5461.50),
    (204, 5461.50),
    (205, 5461.50),
    (206, 5461.50),
    (207, 5461.50),
    (208, 21845.50),
    (209, 21845.50),
    (210, 21845.50),
    (211, 21845.50),
    (212, 21845.50),
    (213, 21845.50),
    (214, 21845.50),
    (215, 21845.50),
    (216, 87381.50),
    (217, 87381.50),
    (218, 87381.50),
    (219, 87381.50),
    (220, 87381.50),
    (221, 87381.50),
    (222, 87381.50),
    (223, 87381.50),
    (224, 349525.50),
    (225, 349525.50),
    (226, 349525.50),
    (227, 349525.50),
    (228, 349525.50),
    (229, 349525.50),
    (230, 349525.50),
    (231, 349525.50),
    (232, 1398101.50),
    (233, 1398101.50),
    (234, 1398101.50),
    (235, 1398101.50),
    (236, 1398101.50),
    (237, 1398101.50),
    (238, 1398101.50),
    (239, 1398101.50),
    (240, 5592405.50),
    (241, 5592405.50),
    (242, 5592405.50),
    (243, 5592405.50),
    (244, 5592405.50),
    (245, 5592405.50),
    (246, 5592405.50),
    (247, 5592405.50),
    (248, 22369621.50),
    (249, 22369621.50),
    (250, 22369621.50),
    (251, 22369621.50),
    (252, 22369621.50),
    (253, 22369621.50),
    (254, 22369621.50),
    (255, 60397977600.00)
])
def test_variance_143(values):
    compressed, variance = values
    _, res = decompress(compressed, s=1, k=4, m=3, return_variance=True)
    assert res == variance
