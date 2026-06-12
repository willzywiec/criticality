Hot Box 8x4x6 (measured density)
C  Cell cards
1 0 -99 lat=1 u=1 imp:n=1
      fill=0:7 0:3 0:5
         2  100    2  101    2  102    2  103  $ row j=0 layer k=0
       104    2  105    2  106    2  107    2  $ row j=1 layer k=0
         2  108    2  109    2  110    2  111  $ row j=2 layer k=0
       112    2  113    2  114    2  115    2  $ row j=3 layer k=0
       116    2  117    2  118    2  119    2  $ row j=0 layer k=1
         2  120    2  121    2  122    2  123  $ row j=1 layer k=1
       124    2  125    2  126    2  127    2  $ row j=2 layer k=1
         2  128    2  129    2  130    2  131  $ row j=3 layer k=1
         2  132    2  133    2  134    2  135  $ row j=0 layer k=2
       136    2  137    2  138    2  139    2  $ row j=1 layer k=2
         2  140    2  141    2  142    2  143  $ row j=2 layer k=2
       144    2  145    2  146    2  147    2  $ row j=3 layer k=2
       148    2  149    2  150    2  151    2  $ row j=0 layer k=3
         2  152    2  153    2  154    2  155  $ row j=1 layer k=3
       156    2  157    2  158    2  159    2  $ row j=2 layer k=3
         2  160    2  161    2  162    2  163  $ row j=3 layer k=3
         2  164    2  165    2  166    2  167  $ row j=0 layer k=4
       168    2  169    2  170    2  171    2  $ row j=1 layer k=4
         2  172    2  173    2  174    2  175  $ row j=2 layer k=4
       176    2  177    2  178    2  179    2  $ row j=3 layer k=4
       180    2  181    2  182    2  183    2  $ row j=0 layer k=5
         2  184    2  185    2  186    2  187  $ row j=1 layer k=5
       188    2  189    2  190    2  191    2  $ row j=2 layer k=5
         2  192    2  193    2  194    2  195  $ row j=3 layer k=5
2 0 -98 fill=1 imp:n=1  $ assembly container
3 0 98 -999 imp:n=1  $ world void
4 0 999 imp:n=0  $ graveyard
C  === void (placeholder) universe ===
500 4 -2.7338  -20  u=2 imp:n=1  $ 1in graphite
501 4 -2.7338  -21  u=2 imp:n=1  $ 2in graphite
502 4 -2.7338  -30  u=2 imp:n=1  $ left rail
503 4 -2.7338  -31  u=2 imp:n=1  $ right rail
504 0       -22  u=2 imp:n=1  $ void slab
505 0       -23  u=2 imp:n=1  $ void top gap
C  === fuel universe 100 slot (1, 2, 1) ===
1000 1 -20.8096  -10        u=100 imp:n=1  $ right foil
1001 2 -20.5359  -11        u=100 imp:n=1  $ left foil
1002 3 -7.1758  -13        u=100 imp:n=1  $ bottom clad
1003 3 -7.1758  -14        u=100 imp:n=1  $ top clad
1004 0             -15 10 11  u=100 imp:n=1  $ foil-plane void
1005 0             -22 12     u=100 imp:n=1  $ void around clad
1006 4 -2.7338  -20        u=100 imp:n=1  $ 1in graphite
1007 4 -2.7338  -21        u=100 imp:n=1  $ 2in graphite
1008 4 -2.7338  -30        u=100 imp:n=1  $ left rail
1009 4 -2.7338  -31        u=100 imp:n=1  $ right rail
1010 0            -23        u=100 imp:n=1  $ top gap
C  === fuel universe 101 slot (1, 4, 1) ===
1011 1 -20.7336  -10        u=101 imp:n=1  $ right foil
1012 2 -21.053  -11        u=101 imp:n=1  $ left foil
1013 3 -7.2137  -13        u=101 imp:n=1  $ bottom clad
1014 3 -7.2137  -14        u=101 imp:n=1  $ top clad
1015 0             -15 10 11  u=101 imp:n=1  $ foil-plane void
1016 0             -22 12     u=101 imp:n=1  $ void around clad
1017 4 -2.7338  -20        u=101 imp:n=1  $ 1in graphite
1018 4 -2.7338  -21        u=101 imp:n=1  $ 2in graphite
1019 4 -2.7338  -30        u=101 imp:n=1  $ left rail
1020 4 -2.7338  -31        u=101 imp:n=1  $ right rail
1021 0            -23        u=101 imp:n=1  $ top gap
C  === fuel universe 102 slot (1, 6, 1) ===
1022 1 -17.5583  -10        u=102 imp:n=1  $ right foil
1023 2 -17.4671  -11        u=102 imp:n=1  $ left foil
1024 3 -7.2707  -13        u=102 imp:n=1  $ bottom clad
1025 3 -7.2707  -14        u=102 imp:n=1  $ top clad
1026 0             -15 10 11  u=102 imp:n=1  $ foil-plane void
1027 0             -22 12     u=102 imp:n=1  $ void around clad
1028 4 -2.7338  -20        u=102 imp:n=1  $ 1in graphite
1029 4 -2.7338  -21        u=102 imp:n=1  $ 2in graphite
1030 4 -2.7338  -30        u=102 imp:n=1  $ left rail
1031 4 -2.7338  -31        u=102 imp:n=1  $ right rail
1032 0            -23        u=102 imp:n=1  $ top gap
C  === fuel universe 103 slot (1, 8, 1) (no measured data -> nominal) ===
1033 1 -18.5835  -10        u=103 imp:n=1  $ right foil
1034 2 -18.5835  -11        u=103 imp:n=1  $ left foil
1035 3 -7.2694  -13        u=103 imp:n=1  $ bottom clad
1036 3 -7.2694  -14        u=103 imp:n=1  $ top clad
1037 0             -15 10 11  u=103 imp:n=1  $ foil-plane void
1038 0             -22 12     u=103 imp:n=1  $ void around clad
1039 4 -2.7338  -20        u=103 imp:n=1  $ 1in graphite
1040 4 -2.7338  -21        u=103 imp:n=1  $ 2in graphite
1041 4 -2.7338  -30        u=103 imp:n=1  $ left rail
1042 4 -2.7338  -31        u=103 imp:n=1  $ right rail
1043 0            -23        u=103 imp:n=1  $ top gap
C  === fuel universe 104 slot (1, 1, 2) ===
1044 1 -17.8777  -10        u=104 imp:n=1  $ right foil
1045 2 -17.8321  -11        u=104 imp:n=1  $ left foil
1046 3 -7.2707  -13        u=104 imp:n=1  $ bottom clad
1047 3 -7.2707  -14        u=104 imp:n=1  $ top clad
1048 0             -15 10 11  u=104 imp:n=1  $ foil-plane void
1049 0             -22 12     u=104 imp:n=1  $ void around clad
1050 4 -2.7338  -20        u=104 imp:n=1  $ 1in graphite
1051 4 -2.7338  -21        u=104 imp:n=1  $ 2in graphite
1052 4 -2.7338  -30        u=104 imp:n=1  $ left rail
1053 4 -2.7338  -31        u=104 imp:n=1  $ right rail
1054 0            -23        u=104 imp:n=1  $ top gap
C  === fuel universe 105 slot (1, 3, 2) ===
1055 1 -17.8016  -10        u=105 imp:n=1  $ right foil
1056 2 -17.9233  -11        u=105 imp:n=1  $ left foil
1057 3 -7.3468  -13        u=105 imp:n=1  $ bottom clad
1058 3 -7.3468  -14        u=105 imp:n=1  $ top clad
1059 0             -15 10 11  u=105 imp:n=1  $ foil-plane void
1060 0             -22 12     u=105 imp:n=1  $ void around clad
1061 4 -2.7338  -20        u=105 imp:n=1  $ 1in graphite
1062 4 -2.7338  -21        u=105 imp:n=1  $ 2in graphite
1063 4 -2.7338  -30        u=105 imp:n=1  $ left rail
1064 4 -2.7338  -31        u=105 imp:n=1  $ right rail
1065 0            -23        u=105 imp:n=1  $ top gap
C  === fuel universe 106 slot (1, 5, 2) ===
1066 1 -17.8168  -10        u=106 imp:n=1  $ right foil
1067 2 -17.9537  -11        u=106 imp:n=1  $ left foil
1068 3 -7.4744  -13        u=106 imp:n=1  $ bottom clad
1069 3 -7.4744  -14        u=106 imp:n=1  $ top clad
1070 0             -15 10 11  u=106 imp:n=1  $ foil-plane void
1071 0             -22 12     u=106 imp:n=1  $ void around clad
1072 4 -2.7338  -20        u=106 imp:n=1  $ 1in graphite
1073 4 -2.7338  -21        u=106 imp:n=1  $ 2in graphite
1074 4 -2.7338  -30        u=106 imp:n=1  $ left rail
1075 4 -2.7338  -31        u=106 imp:n=1  $ right rail
1076 0            -23        u=106 imp:n=1  $ top gap
C  === fuel universe 107 slot (1, 7, 2) ===
1077 1 -20.1405  -10        u=107 imp:n=1  $ right foil
1078 2 -20.4142  -11        u=107 imp:n=1  $ left foil
1079 3 -7.2707  -13        u=107 imp:n=1  $ bottom clad
1080 3 -7.2707  -14        u=107 imp:n=1  $ top clad
1081 0             -15 10 11  u=107 imp:n=1  $ foil-plane void
1082 0             -22 12     u=107 imp:n=1  $ void around clad
1083 4 -2.7338  -20        u=107 imp:n=1  $ 1in graphite
1084 4 -2.7338  -21        u=107 imp:n=1  $ 2in graphite
1085 4 -2.7338  -30        u=107 imp:n=1  $ left rail
1086 4 -2.7338  -31        u=107 imp:n=1  $ right rail
1087 0            -23        u=107 imp:n=1  $ top gap
C  === fuel universe 108 slot (1, 2, 3) ===
1088 1 -19.0608  -10        u=108 imp:n=1  $ right foil
1089 2 -19.1672  -11        u=108 imp:n=1  $ left foil
1090 3 -7.1241  -13        u=108 imp:n=1  $ bottom clad
1091 3 -7.1241  -14        u=108 imp:n=1  $ top clad
1092 0             -15 10 11  u=108 imp:n=1  $ foil-plane void
1093 0             -22 12     u=108 imp:n=1  $ void around clad
1094 4 -2.7338  -20        u=108 imp:n=1  $ 1in graphite
1095 4 -2.7338  -21        u=108 imp:n=1  $ 2in graphite
1096 4 -2.7338  -30        u=108 imp:n=1  $ left rail
1097 4 -2.7338  -31        u=108 imp:n=1  $ right rail
1098 0            -23        u=108 imp:n=1  $ top gap
C  === fuel universe 109 slot (1, 4, 3) ===
1099 1 -19.0  -10        u=109 imp:n=1  $ right foil
1100 2 -19.0  -11        u=109 imp:n=1  $ left foil
1101 3 -7.3137  -13        u=109 imp:n=1  $ bottom clad
1102 3 -7.3137  -14        u=109 imp:n=1  $ top clad
1103 0             -15 10 11  u=109 imp:n=1  $ foil-plane void
1104 0             -22 12     u=109 imp:n=1  $ void around clad
1105 4 -2.7338  -20        u=109 imp:n=1  $ 1in graphite
1106 4 -2.7338  -21        u=109 imp:n=1  $ 2in graphite
1107 4 -2.7338  -30        u=109 imp:n=1  $ left rail
1108 4 -2.7338  -31        u=109 imp:n=1  $ right rail
1109 0            -23        u=109 imp:n=1  $ top gap
C  === fuel universe 110 slot (1, 6, 3) ===
1110 1 -19.1977  -10        u=110 imp:n=1  $ right foil
1111 2 -19.0182  -11        u=110 imp:n=1  $ left foil
1112 3 -7.7475  -13        u=110 imp:n=1  $ bottom clad
1113 3 -7.7475  -14        u=110 imp:n=1  $ top clad
1114 0             -15 10 11  u=110 imp:n=1  $ foil-plane void
1115 0             -22 12     u=110 imp:n=1  $ void around clad
1116 4 -2.7338  -20        u=110 imp:n=1  $ 1in graphite
1117 4 -2.7338  -21        u=110 imp:n=1  $ 2in graphite
1118 4 -2.7338  -30        u=110 imp:n=1  $ left rail
1119 4 -2.7338  -31        u=110 imp:n=1  $ right rail
1120 0            -23        u=110 imp:n=1  $ top gap
C  === fuel universe 111 slot (1, 8, 3) ===
1121 1 -19.3497  -10        u=111 imp:n=1  $ right foil
1122 2 -18.7445  -11        u=111 imp:n=1  $ left foil
1123 3 -7.2923  -13        u=111 imp:n=1  $ bottom clad
1124 3 -7.2923  -14        u=111 imp:n=1  $ top clad
1125 0             -15 10 11  u=111 imp:n=1  $ foil-plane void
1126 0             -22 12     u=111 imp:n=1  $ void around clad
1127 4 -2.7338  -20        u=111 imp:n=1  $ 1in graphite
1128 4 -2.7338  -21        u=111 imp:n=1  $ 2in graphite
1129 4 -2.7338  -30        u=111 imp:n=1  $ left rail
1130 4 -2.7338  -31        u=111 imp:n=1  $ right rail
1131 0            -23        u=111 imp:n=1  $ top gap
C  === fuel universe 112 slot (1, 1, 4) ===
1132 1 -17.8777  -10        u=112 imp:n=1  $ right foil
1133 2 -17.8321  -11        u=112 imp:n=1  $ left foil
1134 3 -7.2707  -13        u=112 imp:n=1  $ bottom clad
1135 3 -7.2707  -14        u=112 imp:n=1  $ top clad
1136 0             -15 10 11  u=112 imp:n=1  $ foil-plane void
1137 0             -22 12     u=112 imp:n=1  $ void around clad
1138 4 -2.7338  -20        u=112 imp:n=1  $ 1in graphite
1139 4 -2.7338  -21        u=112 imp:n=1  $ 2in graphite
1140 4 -2.7338  -30        u=112 imp:n=1  $ left rail
1141 4 -2.7338  -31        u=112 imp:n=1  $ right rail
1142 0            -23        u=112 imp:n=1  $ top gap
C  === fuel universe 113 slot (1, 3, 4) ===
1143 1 -17.8016  -10        u=113 imp:n=1  $ right foil
1144 2 -17.9233  -11        u=113 imp:n=1  $ left foil
1145 3 -7.3468  -13        u=113 imp:n=1  $ bottom clad
1146 3 -7.3468  -14        u=113 imp:n=1  $ top clad
1147 0             -15 10 11  u=113 imp:n=1  $ foil-plane void
1148 0             -22 12     u=113 imp:n=1  $ void around clad
1149 4 -2.7338  -20        u=113 imp:n=1  $ 1in graphite
1150 4 -2.7338  -21        u=113 imp:n=1  $ 2in graphite
1151 4 -2.7338  -30        u=113 imp:n=1  $ left rail
1152 4 -2.7338  -31        u=113 imp:n=1  $ right rail
1153 0            -23        u=113 imp:n=1  $ top gap
C  === fuel universe 114 slot (1, 5, 4) ===
1154 1 -17.8168  -10        u=114 imp:n=1  $ right foil
1155 2 -17.9537  -11        u=114 imp:n=1  $ left foil
1156 3 -7.4744  -13        u=114 imp:n=1  $ bottom clad
1157 3 -7.4744  -14        u=114 imp:n=1  $ top clad
1158 0             -15 10 11  u=114 imp:n=1  $ foil-plane void
1159 0             -22 12     u=114 imp:n=1  $ void around clad
1160 4 -2.7338  -20        u=114 imp:n=1  $ 1in graphite
1161 4 -2.7338  -21        u=114 imp:n=1  $ 2in graphite
1162 4 -2.7338  -30        u=114 imp:n=1  $ left rail
1163 4 -2.7338  -31        u=114 imp:n=1  $ right rail
1164 0            -23        u=114 imp:n=1  $ top gap
C  === fuel universe 115 slot (1, 7, 4) ===
1165 1 -20.1405  -10        u=115 imp:n=1  $ right foil
1166 2 -20.4142  -11        u=115 imp:n=1  $ left foil
1167 3 -7.2707  -13        u=115 imp:n=1  $ bottom clad
1168 3 -7.2707  -14        u=115 imp:n=1  $ top clad
1169 0             -15 10 11  u=115 imp:n=1  $ foil-plane void
1170 0             -22 12     u=115 imp:n=1  $ void around clad
1171 4 -2.7338  -20        u=115 imp:n=1  $ 1in graphite
1172 4 -2.7338  -21        u=115 imp:n=1  $ 2in graphite
1173 4 -2.7338  -30        u=115 imp:n=1  $ left rail
1174 4 -2.7338  -31        u=115 imp:n=1  $ right rail
1175 0            -23        u=115 imp:n=1  $ top gap
C  === fuel universe 116 slot (2, 1, 1) ===
1176 1 -18.0297  -10        u=116 imp:n=1  $ right foil
1177 2 -17.9537  -11        u=116 imp:n=1  $ left foil
1178 3 -7.2882  -13        u=116 imp:n=1  $ bottom clad
1179 3 -7.2882  -14        u=116 imp:n=1  $ top clad
1180 0             -15 10 11  u=116 imp:n=1  $ foil-plane void
1181 0             -22 12     u=116 imp:n=1  $ void around clad
1182 4 -2.7338  -20        u=116 imp:n=1  $ 1in graphite
1183 4 -2.7338  -21        u=116 imp:n=1  $ 2in graphite
1184 4 -2.7338  -30        u=116 imp:n=1  $ left rail
1185 4 -2.7338  -31        u=116 imp:n=1  $ right rail
1186 0            -23        u=116 imp:n=1  $ top gap
C  === fuel universe 117 slot (2, 3, 1) ===
1187 1 -18.638  -10        u=117 imp:n=1  $ right foil
1188 2 -17.3606  -11        u=117 imp:n=1  $ left foil
1189 3 -7.3537  -13        u=117 imp:n=1  $ bottom clad
1190 3 -7.3537  -14        u=117 imp:n=1  $ top clad
1191 0             -15 10 11  u=117 imp:n=1  $ foil-plane void
1192 0             -22 12     u=117 imp:n=1  $ void around clad
1193 4 -2.7338  -20        u=117 imp:n=1  $ 1in graphite
1194 4 -2.7338  -21        u=117 imp:n=1  $ 2in graphite
1195 4 -2.7338  -30        u=117 imp:n=1  $ left rail
1196 4 -2.7338  -31        u=117 imp:n=1  $ right rail
1197 0            -23        u=117 imp:n=1  $ top gap
C  === fuel universe 118 slot (2, 5, 1) ===
1198 1 -17.8625  -10        u=118 imp:n=1  $ right foil
1199 2 -17.9385  -11        u=118 imp:n=1  $ left foil
1200 3 -7.3985  -13        u=118 imp:n=1  $ bottom clad
1201 3 -7.3985  -14        u=118 imp:n=1  $ top clad
1202 0             -15 10 11  u=118 imp:n=1  $ foil-plane void
1203 0             -22 12     u=118 imp:n=1  $ void around clad
1204 4 -2.7338  -20        u=118 imp:n=1  $ 1in graphite
1205 4 -2.7338  -21        u=118 imp:n=1  $ 2in graphite
1206 4 -2.7338  -30        u=118 imp:n=1  $ left rail
1207 4 -2.7338  -31        u=118 imp:n=1  $ right rail
1208 0            -23        u=118 imp:n=1  $ top gap
C  === fuel universe 119 slot (2, 7, 1) ===
1209 1 -18.0297  -10        u=119 imp:n=1  $ right foil
1210 2 -17.9993  -11        u=119 imp:n=1  $ left foil
1211 3 -7.3468  -13        u=119 imp:n=1  $ bottom clad
1212 3 -7.3468  -14        u=119 imp:n=1  $ top clad
1213 0             -15 10 11  u=119 imp:n=1  $ foil-plane void
1214 0             -22 12     u=119 imp:n=1  $ void around clad
1215 4 -2.7338  -20        u=119 imp:n=1  $ 1in graphite
1216 4 -2.7338  -21        u=119 imp:n=1  $ 2in graphite
1217 4 -2.7338  -30        u=119 imp:n=1  $ left rail
1218 4 -2.7338  -31        u=119 imp:n=1  $ right rail
1219 0            -23        u=119 imp:n=1  $ top gap
C  === fuel universe 120 slot (2, 2, 2) ===
1220 1 -18.4708  -10        u=120 imp:n=1  $ right foil
1221 2 -18.3491  -11        u=120 imp:n=1  $ left foil
1222 3 -7.2365  -13        u=120 imp:n=1  $ bottom clad
1223 3 -7.2365  -14        u=120 imp:n=1  $ top clad
1224 0             -15 10 11  u=120 imp:n=1  $ foil-plane void
1225 0             -22 12     u=120 imp:n=1  $ void around clad
1226 4 -2.7338  -20        u=120 imp:n=1  $ 1in graphite
1227 4 -2.7338  -21        u=120 imp:n=1  $ 2in graphite
1228 4 -2.7338  -30        u=120 imp:n=1  $ left rail
1229 4 -2.7338  -31        u=120 imp:n=1  $ right rail
1230 0            -23        u=120 imp:n=1  $ top gap
C  === fuel universe 121 slot (2, 4, 2) ===
1231 1 -18.4555  -10        u=121 imp:n=1  $ right foil
1232 2 -18.4099  -11        u=121 imp:n=1  $ left foil
1233 3 -7.2261  -13        u=121 imp:n=1  $ bottom clad
1234 3 -7.2261  -14        u=121 imp:n=1  $ top clad
1235 0             -15 10 11  u=121 imp:n=1  $ foil-plane void
1236 0             -22 12     u=121 imp:n=1  $ void around clad
1237 4 -2.7338  -20        u=121 imp:n=1  $ 1in graphite
1238 4 -2.7338  -21        u=121 imp:n=1  $ 2in graphite
1239 4 -2.7338  -30        u=121 imp:n=1  $ left rail
1240 4 -2.7338  -31        u=121 imp:n=1  $ right rail
1241 0            -23        u=121 imp:n=1  $ top gap
C  === fuel universe 122 slot (2, 6, 2) ===
1242 1 -18.3947  -10        u=122 imp:n=1  $ right foil
1243 2 -18.3339  -11        u=122 imp:n=1  $ left foil
1244 3 -7.2707  -13        u=122 imp:n=1  $ bottom clad
1245 3 -7.2707  -14        u=122 imp:n=1  $ top clad
1246 0             -15 10 11  u=122 imp:n=1  $ foil-plane void
1247 0             -22 12     u=122 imp:n=1  $ void around clad
1248 4 -2.7338  -20        u=122 imp:n=1  $ 1in graphite
1249 4 -2.7338  -21        u=122 imp:n=1  $ 2in graphite
1250 4 -2.7338  -30        u=122 imp:n=1  $ left rail
1251 4 -2.7338  -31        u=122 imp:n=1  $ right rail
1252 0            -23        u=122 imp:n=1  $ top gap
C  === fuel universe 123 slot (2, 8, 2) ===
1253 1 -18.1666  -10        u=123 imp:n=1  $ right foil
1254 2 -18.1058  -11        u=123 imp:n=1  $ left foil
1255 3 -7.2707  -13        u=123 imp:n=1  $ bottom clad
1256 3 -7.2707  -14        u=123 imp:n=1  $ top clad
1257 0             -15 10 11  u=123 imp:n=1  $ foil-plane void
1258 0             -22 12     u=123 imp:n=1  $ void around clad
1259 4 -2.7338  -20        u=123 imp:n=1  $ 1in graphite
1260 4 -2.7338  -21        u=123 imp:n=1  $ 2in graphite
1261 4 -2.7338  -30        u=123 imp:n=1  $ left rail
1262 4 -2.7338  -31        u=123 imp:n=1  $ right rail
1263 0            -23        u=123 imp:n=1  $ top gap
C  === fuel universe 124 slot (2, 1, 3) ===
1264 1 -17.8777  -10        u=124 imp:n=1  $ right foil
1265 2 -18.0602  -11        u=124 imp:n=1  $ left foil
1266 3 -7.4365  -13        u=124 imp:n=1  $ bottom clad
1267 3 -7.4365  -14        u=124 imp:n=1  $ top clad
1268 0             -15 10 11  u=124 imp:n=1  $ foil-plane void
1269 0             -22 12     u=124 imp:n=1  $ void around clad
1270 4 -2.7338  -20        u=124 imp:n=1  $ 1in graphite
1271 4 -2.7338  -21        u=124 imp:n=1  $ 2in graphite
1272 4 -2.7338  -30        u=124 imp:n=1  $ left rail
1273 4 -2.7338  -31        u=124 imp:n=1  $ right rail
1274 0            -23        u=124 imp:n=1  $ top gap
C  === fuel universe 125 slot (2, 3, 3) ===
1275 1 -18.0754  -10        u=125 imp:n=1  $ right foil
1276 2 -18.0145  -11        u=125 imp:n=1  $ left foil
1277 3 -7.402  -13        u=125 imp:n=1  $ bottom clad
1278 3 -7.402  -14        u=125 imp:n=1  $ top clad
1279 0             -15 10 11  u=125 imp:n=1  $ foil-plane void
1280 0             -22 12     u=125 imp:n=1  $ void around clad
1281 4 -2.7338  -20        u=125 imp:n=1  $ 1in graphite
1282 4 -2.7338  -21        u=125 imp:n=1  $ 2in graphite
1283 4 -2.7338  -30        u=125 imp:n=1  $ left rail
1284 4 -2.7338  -31        u=125 imp:n=1  $ right rail
1285 0            -23        u=125 imp:n=1  $ top gap
C  === fuel universe 126 slot (2, 5, 3) ===
1286 1 -17.9385  -10        u=126 imp:n=1  $ right foil
1287 2 -17.9537  -11        u=126 imp:n=1  $ left foil
1288 3 -7.171  -13        u=126 imp:n=1  $ bottom clad
1289 3 -7.171  -14        u=126 imp:n=1  $ top clad
1290 0             -15 10 11  u=126 imp:n=1  $ foil-plane void
1291 0             -22 12     u=126 imp:n=1  $ void around clad
1292 4 -2.7338  -20        u=126 imp:n=1  $ 1in graphite
1293 4 -2.7338  -21        u=126 imp:n=1  $ 2in graphite
1294 4 -2.7338  -30        u=126 imp:n=1  $ left rail
1295 4 -2.7338  -31        u=126 imp:n=1  $ right rail
1296 0            -23        u=126 imp:n=1  $ top gap
C  === fuel universe 127 slot (2, 7, 3) ===
1297 1 -17.6648  -10        u=127 imp:n=1  $ right foil
1298 2 -17.8321  -11        u=127 imp:n=1  $ left foil
1299 3 -7.2707  -13        u=127 imp:n=1  $ bottom clad
1300 3 -7.2707  -14        u=127 imp:n=1  $ top clad
1301 0             -15 10 11  u=127 imp:n=1  $ foil-plane void
1302 0             -22 12     u=127 imp:n=1  $ void around clad
1303 4 -2.7338  -20        u=127 imp:n=1  $ 1in graphite
1304 4 -2.7338  -21        u=127 imp:n=1  $ 2in graphite
1305 4 -2.7338  -30        u=127 imp:n=1  $ left rail
1306 4 -2.7338  -31        u=127 imp:n=1  $ right rail
1307 0            -23        u=127 imp:n=1  $ top gap
C  === fuel universe 128 slot (2, 2, 4) ===
1308 1 -18.4708  -10        u=128 imp:n=1  $ right foil
1309 2 -18.3491  -11        u=128 imp:n=1  $ left foil
1310 3 -7.2365  -13        u=128 imp:n=1  $ bottom clad
1311 3 -7.2365  -14        u=128 imp:n=1  $ top clad
1312 0             -15 10 11  u=128 imp:n=1  $ foil-plane void
1313 0             -22 12     u=128 imp:n=1  $ void around clad
1314 4 -2.7338  -20        u=128 imp:n=1  $ 1in graphite
1315 4 -2.7338  -21        u=128 imp:n=1  $ 2in graphite
1316 4 -2.7338  -30        u=128 imp:n=1  $ left rail
1317 4 -2.7338  -31        u=128 imp:n=1  $ right rail
1318 0            -23        u=128 imp:n=1  $ top gap
C  === fuel universe 129 slot (2, 4, 4) ===
1319 1 -18.4555  -10        u=129 imp:n=1  $ right foil
1320 2 -18.4099  -11        u=129 imp:n=1  $ left foil
1321 3 -7.2261  -13        u=129 imp:n=1  $ bottom clad
1322 3 -7.2261  -14        u=129 imp:n=1  $ top clad
1323 0             -15 10 11  u=129 imp:n=1  $ foil-plane void
1324 0             -22 12     u=129 imp:n=1  $ void around clad
1325 4 -2.7338  -20        u=129 imp:n=1  $ 1in graphite
1326 4 -2.7338  -21        u=129 imp:n=1  $ 2in graphite
1327 4 -2.7338  -30        u=129 imp:n=1  $ left rail
1328 4 -2.7338  -31        u=129 imp:n=1  $ right rail
1329 0            -23        u=129 imp:n=1  $ top gap
C  === fuel universe 130 slot (2, 6, 4) ===
1330 1 -18.3947  -10        u=130 imp:n=1  $ right foil
1331 2 -18.3339  -11        u=130 imp:n=1  $ left foil
1332 3 -7.2707  -13        u=130 imp:n=1  $ bottom clad
1333 3 -7.2707  -14        u=130 imp:n=1  $ top clad
1334 0             -15 10 11  u=130 imp:n=1  $ foil-plane void
1335 0             -22 12     u=130 imp:n=1  $ void around clad
1336 4 -2.7338  -20        u=130 imp:n=1  $ 1in graphite
1337 4 -2.7338  -21        u=130 imp:n=1  $ 2in graphite
1338 4 -2.7338  -30        u=130 imp:n=1  $ left rail
1339 4 -2.7338  -31        u=130 imp:n=1  $ right rail
1340 0            -23        u=130 imp:n=1  $ top gap
C  === fuel universe 131 slot (2, 8, 4) ===
1341 1 -18.1666  -10        u=131 imp:n=1  $ right foil
1342 2 -18.1058  -11        u=131 imp:n=1  $ left foil
1343 3 -7.2707  -13        u=131 imp:n=1  $ bottom clad
1344 3 -7.2707  -14        u=131 imp:n=1  $ top clad
1345 0             -15 10 11  u=131 imp:n=1  $ foil-plane void
1346 0             -22 12     u=131 imp:n=1  $ void around clad
1347 4 -2.7338  -20        u=131 imp:n=1  $ 1in graphite
1348 4 -2.7338  -21        u=131 imp:n=1  $ 2in graphite
1349 4 -2.7338  -30        u=131 imp:n=1  $ left rail
1350 4 -2.7338  -31        u=131 imp:n=1  $ right rail
1351 0            -23        u=131 imp:n=1  $ top gap
C  === fuel universe 132 slot (3, 2, 1) ===
1352 1 -18.1362  -10        u=132 imp:n=1  $ right foil
1353 2 -18.2122  -11        u=132 imp:n=1  $ left foil
1354 3 -7.3434  -13        u=132 imp:n=1  $ bottom clad
1355 3 -7.3434  -14        u=132 imp:n=1  $ top clad
1356 0             -15 10 11  u=132 imp:n=1  $ foil-plane void
1357 0             -22 12     u=132 imp:n=1  $ void around clad
1358 4 -2.7338  -20        u=132 imp:n=1  $ 1in graphite
1359 4 -2.7338  -21        u=132 imp:n=1  $ 2in graphite
1360 4 -2.7338  -30        u=132 imp:n=1  $ left rail
1361 4 -2.7338  -31        u=132 imp:n=1  $ right rail
1362 0            -23        u=132 imp:n=1  $ top gap
C  === fuel universe 133 slot (3, 4, 1) ===
1363 1 -17.9689  -10        u=133 imp:n=1  $ right foil
1364 2 -18.1362  -11        u=133 imp:n=1  $ left foil
1365 3 -7.1227  -13        u=133 imp:n=1  $ bottom clad
1366 3 -7.1227  -14        u=133 imp:n=1  $ top clad
1367 0             -15 10 11  u=133 imp:n=1  $ foil-plane void
1368 0             -22 12     u=133 imp:n=1  $ void around clad
1369 4 -2.7338  -20        u=133 imp:n=1  $ 1in graphite
1370 4 -2.7338  -21        u=133 imp:n=1  $ 2in graphite
1371 4 -2.7338  -30        u=133 imp:n=1  $ left rail
1372 4 -2.7338  -31        u=133 imp:n=1  $ right rail
1373 0            -23        u=133 imp:n=1  $ top gap
C  === fuel universe 134 slot (3, 6, 1) ===
1374 1 -18.1514  -10        u=134 imp:n=1  $ right foil
1375 2 -18.2274  -11        u=134 imp:n=1  $ left foil
1376 3 -7.1296  -13        u=134 imp:n=1  $ bottom clad
1377 3 -7.1296  -14        u=134 imp:n=1  $ top clad
1378 0             -15 10 11  u=134 imp:n=1  $ foil-plane void
1379 0             -22 12     u=134 imp:n=1  $ void around clad
1380 4 -2.7338  -20        u=134 imp:n=1  $ 1in graphite
1381 4 -2.7338  -21        u=134 imp:n=1  $ 2in graphite
1382 4 -2.7338  -30        u=134 imp:n=1  $ left rail
1383 4 -2.7338  -31        u=134 imp:n=1  $ right rail
1384 0            -23        u=134 imp:n=1  $ top gap
C  === fuel universe 135 slot (3, 8, 1) ===
1385 1 -17.8321  -10        u=135 imp:n=1  $ right foil
1386 2 -18.1666  -11        u=135 imp:n=1  $ left foil
1387 3 -7.3537  -13        u=135 imp:n=1  $ bottom clad
1388 3 -7.3537  -14        u=135 imp:n=1  $ top clad
1389 0             -15 10 11  u=135 imp:n=1  $ foil-plane void
1390 0             -22 12     u=135 imp:n=1  $ void around clad
1391 4 -2.7338  -20        u=135 imp:n=1  $ 1in graphite
1392 4 -2.7338  -21        u=135 imp:n=1  $ 2in graphite
1393 4 -2.7338  -30        u=135 imp:n=1  $ left rail
1394 4 -2.7338  -31        u=135 imp:n=1  $ right rail
1395 0            -23        u=135 imp:n=1  $ top gap
C  === fuel universe 136 slot (3, 1, 2) ===
1396 1 -18.2883  -10        u=136 imp:n=1  $ right foil
1397 2 -18.1514  -11        u=136 imp:n=1  $ left foil
1398 3 -7.3227  -13        u=136 imp:n=1  $ bottom clad
1399 3 -7.3227  -14        u=136 imp:n=1  $ top clad
1400 0             -15 10 11  u=136 imp:n=1  $ foil-plane void
1401 0             -22 12     u=136 imp:n=1  $ void around clad
1402 4 -2.7338  -20        u=136 imp:n=1  $ 1in graphite
1403 4 -2.7338  -21        u=136 imp:n=1  $ 2in graphite
1404 4 -2.7338  -30        u=136 imp:n=1  $ left rail
1405 4 -2.7338  -31        u=136 imp:n=1  $ right rail
1406 0            -23        u=136 imp:n=1  $ top gap
C  === fuel universe 137 slot (3, 3, 2) ===
1407 1 -17.8016  -10        u=137 imp:n=1  $ right foil
1408 2 -18.6989  -11        u=137 imp:n=1  $ left foil
1409 3 -7.2707  -13        u=137 imp:n=1  $ bottom clad
1410 3 -7.2707  -14        u=137 imp:n=1  $ top clad
1411 0             -15 10 11  u=137 imp:n=1  $ foil-plane void
1412 0             -22 12     u=137 imp:n=1  $ void around clad
1413 4 -2.7338  -20        u=137 imp:n=1  $ 1in graphite
1414 4 -2.7338  -21        u=137 imp:n=1  $ 2in graphite
1415 4 -2.7338  -30        u=137 imp:n=1  $ left rail
1416 4 -2.7338  -31        u=137 imp:n=1  $ right rail
1417 0            -23        u=137 imp:n=1  $ top gap
C  === fuel universe 138 slot (3, 5, 2) ===
1418 1 -17.8168  -10        u=138 imp:n=1  $ right foil
1419 2 -19.0152  -11        u=138 imp:n=1  $ left foil
1420 3 -7.3027  -13        u=138 imp:n=1  $ bottom clad
1421 3 -7.3027  -14        u=138 imp:n=1  $ top clad
1422 0             -15 10 11  u=138 imp:n=1  $ foil-plane void
1423 0             -22 12     u=138 imp:n=1  $ void around clad
1424 4 -2.7338  -20        u=138 imp:n=1  $ 1in graphite
1425 4 -2.7338  -21        u=138 imp:n=1  $ 2in graphite
1426 4 -2.7338  -30        u=138 imp:n=1  $ left rail
1427 4 -2.7338  -31        u=138 imp:n=1  $ right rail
1428 0            -23        u=138 imp:n=1  $ top gap
C  === fuel universe 139 slot (3, 7, 2) ===
1429 1 -18.5316  -10        u=139 imp:n=1  $ right foil
1430 2 -18.2731  -11        u=139 imp:n=1  $ left foil
1431 3 -7.2707  -13        u=139 imp:n=1  $ bottom clad
1432 3 -7.2707  -14        u=139 imp:n=1  $ top clad
1433 0             -15 10 11  u=139 imp:n=1  $ foil-plane void
1434 0             -22 12     u=139 imp:n=1  $ void around clad
1435 4 -2.7338  -20        u=139 imp:n=1  $ 1in graphite
1436 4 -2.7338  -21        u=139 imp:n=1  $ 2in graphite
1437 4 -2.7338  -30        u=139 imp:n=1  $ left rail
1438 4 -2.7338  -31        u=139 imp:n=1  $ right rail
1439 0            -23        u=139 imp:n=1  $ top gap
C  === fuel universe 140 slot (3, 2, 3) ===
1440 1 -17.8016  -10        u=140 imp:n=1  $ right foil
1441 2 -17.7256  -11        u=140 imp:n=1  $ left foil
1442 3 -7.2707  -13        u=140 imp:n=1  $ bottom clad
1443 3 -7.2707  -14        u=140 imp:n=1  $ top clad
1444 0             -15 10 11  u=140 imp:n=1  $ foil-plane void
1445 0             -22 12     u=140 imp:n=1  $ void around clad
1446 4 -2.7338  -20        u=140 imp:n=1  $ 1in graphite
1447 4 -2.7338  -21        u=140 imp:n=1  $ 2in graphite
1448 4 -2.7338  -30        u=140 imp:n=1  $ left rail
1449 4 -2.7338  -31        u=140 imp:n=1  $ right rail
1450 0            -23        u=140 imp:n=1  $ top gap
C  === fuel universe 141 slot (3, 4, 3) ===
1451 1 -18.2579  -10        u=141 imp:n=1  $ right foil
1452 2 -18.121  -11        u=141 imp:n=1  $ left foil
1453 3 -7.1986  -13        u=141 imp:n=1  $ bottom clad
1454 3 -7.1986  -14        u=141 imp:n=1  $ top clad
1455 0             -15 10 11  u=141 imp:n=1  $ foil-plane void
1456 0             -22 12     u=141 imp:n=1  $ void around clad
1457 4 -2.7338  -20        u=141 imp:n=1  $ 1in graphite
1458 4 -2.7338  -21        u=141 imp:n=1  $ 2in graphite
1459 4 -2.7338  -30        u=141 imp:n=1  $ left rail
1460 4 -2.7338  -31        u=141 imp:n=1  $ right rail
1461 0            -23        u=141 imp:n=1  $ top gap
C  === fuel universe 142 slot (3, 6, 3) ===
1462 1 -18.1362  -10        u=142 imp:n=1  $ right foil
1463 2 -18.0145  -11        u=142 imp:n=1  $ left foil
1464 3 -7.2707  -13        u=142 imp:n=1  $ bottom clad
1465 3 -7.2707  -14        u=142 imp:n=1  $ top clad
1466 0             -15 10 11  u=142 imp:n=1  $ foil-plane void
1467 0             -22 12     u=142 imp:n=1  $ void around clad
1468 4 -2.7338  -20        u=142 imp:n=1  $ 1in graphite
1469 4 -2.7338  -21        u=142 imp:n=1  $ 2in graphite
1470 4 -2.7338  -30        u=142 imp:n=1  $ left rail
1471 4 -2.7338  -31        u=142 imp:n=1  $ right rail
1472 0            -23        u=142 imp:n=1  $ top gap
C  === fuel universe 143 slot (3, 8, 3) ===
1473 1 -17.8473  -10        u=143 imp:n=1  $ right foil
1474 2 -17.8777  -11        u=143 imp:n=1  $ left foil
1475 3 -7.2707  -13        u=143 imp:n=1  $ bottom clad
1476 3 -7.2707  -14        u=143 imp:n=1  $ top clad
1477 0             -15 10 11  u=143 imp:n=1  $ foil-plane void
1478 0             -22 12     u=143 imp:n=1  $ void around clad
1479 4 -2.7338  -20        u=143 imp:n=1  $ 1in graphite
1480 4 -2.7338  -21        u=143 imp:n=1  $ 2in graphite
1481 4 -2.7338  -30        u=143 imp:n=1  $ left rail
1482 4 -2.7338  -31        u=143 imp:n=1  $ right rail
1483 0            -23        u=143 imp:n=1  $ top gap
C  === fuel universe 144 slot (3, 1, 4) ===
1484 1 -18.2883  -10        u=144 imp:n=1  $ right foil
1485 2 -18.1514  -11        u=144 imp:n=1  $ left foil
1486 3 -7.3227  -13        u=144 imp:n=1  $ bottom clad
1487 3 -7.3227  -14        u=144 imp:n=1  $ top clad
1488 0             -15 10 11  u=144 imp:n=1  $ foil-plane void
1489 0             -22 12     u=144 imp:n=1  $ void around clad
1490 4 -2.7338  -20        u=144 imp:n=1  $ 1in graphite
1491 4 -2.7338  -21        u=144 imp:n=1  $ 2in graphite
1492 4 -2.7338  -30        u=144 imp:n=1  $ left rail
1493 4 -2.7338  -31        u=144 imp:n=1  $ right rail
1494 0            -23        u=144 imp:n=1  $ top gap
C  === fuel universe 145 slot (3, 3, 4) ===
1495 1 -17.8016  -10        u=145 imp:n=1  $ right foil
1496 2 -18.6989  -11        u=145 imp:n=1  $ left foil
1497 3 -7.2707  -13        u=145 imp:n=1  $ bottom clad
1498 3 -7.2707  -14        u=145 imp:n=1  $ top clad
1499 0             -15 10 11  u=145 imp:n=1  $ foil-plane void
1500 0             -22 12     u=145 imp:n=1  $ void around clad
1501 4 -2.7338  -20        u=145 imp:n=1  $ 1in graphite
1502 4 -2.7338  -21        u=145 imp:n=1  $ 2in graphite
1503 4 -2.7338  -30        u=145 imp:n=1  $ left rail
1504 4 -2.7338  -31        u=145 imp:n=1  $ right rail
1505 0            -23        u=145 imp:n=1  $ top gap
C  === fuel universe 146 slot (3, 5, 4) ===
1506 1 -17.8168  -10        u=146 imp:n=1  $ right foil
1507 2 -19.0152  -11        u=146 imp:n=1  $ left foil
1508 3 -7.3027  -13        u=146 imp:n=1  $ bottom clad
1509 3 -7.3027  -14        u=146 imp:n=1  $ top clad
1510 0             -15 10 11  u=146 imp:n=1  $ foil-plane void
1511 0             -22 12     u=146 imp:n=1  $ void around clad
1512 4 -2.7338  -20        u=146 imp:n=1  $ 1in graphite
1513 4 -2.7338  -21        u=146 imp:n=1  $ 2in graphite
1514 4 -2.7338  -30        u=146 imp:n=1  $ left rail
1515 4 -2.7338  -31        u=146 imp:n=1  $ right rail
1516 0            -23        u=146 imp:n=1  $ top gap
C  === fuel universe 147 slot (3, 7, 4) ===
1517 1 -18.5316  -10        u=147 imp:n=1  $ right foil
1518 2 -18.2731  -11        u=147 imp:n=1  $ left foil
1519 3 -7.2707  -13        u=147 imp:n=1  $ bottom clad
1520 3 -7.2707  -14        u=147 imp:n=1  $ top clad
1521 0             -15 10 11  u=147 imp:n=1  $ foil-plane void
1522 0             -22 12     u=147 imp:n=1  $ void around clad
1523 4 -2.7338  -20        u=147 imp:n=1  $ 1in graphite
1524 4 -2.7338  -21        u=147 imp:n=1  $ 2in graphite
1525 4 -2.7338  -30        u=147 imp:n=1  $ left rail
1526 4 -2.7338  -31        u=147 imp:n=1  $ right rail
1527 0            -23        u=147 imp:n=1  $ top gap
C  === fuel universe 148 slot (4, 1, 1) ===
1528 1 -18.927  -10        u=148 imp:n=1  $ right foil
1529 2 -21.0377  -11        u=148 imp:n=1  $ left foil
1530 3 -7.1441  -13        u=148 imp:n=1  $ bottom clad
1531 3 -7.1441  -14        u=148 imp:n=1  $ top clad
1532 0             -15 10 11  u=148 imp:n=1  $ foil-plane void
1533 0             -22 12     u=148 imp:n=1  $ void around clad
1534 4 -2.7338  -20        u=148 imp:n=1  $ 1in graphite
1535 4 -2.7338  -21        u=148 imp:n=1  $ 2in graphite
1536 4 -2.7338  -30        u=148 imp:n=1  $ left rail
1537 4 -2.7338  -31        u=148 imp:n=1  $ right rail
1538 0            -23        u=148 imp:n=1  $ top gap
C  === fuel universe 149 slot (4, 3, 1) ===
1539 1 -18.8509  -10        u=149 imp:n=1  $ right foil
1540 2 -20.4295  -11        u=149 imp:n=1  $ left foil
1541 3 -7.0924  -13        u=149 imp:n=1  $ bottom clad
1542 3 -7.0924  -14        u=149 imp:n=1  $ top clad
1543 0             -15 10 11  u=149 imp:n=1  $ foil-plane void
1544 0             -22 12     u=149 imp:n=1  $ void around clad
1545 4 -2.7338  -20        u=149 imp:n=1  $ 1in graphite
1546 4 -2.7338  -21        u=149 imp:n=1  $ 2in graphite
1547 4 -2.7338  -30        u=149 imp:n=1  $ left rail
1548 4 -2.7338  -31        u=149 imp:n=1  $ right rail
1549 0            -23        u=149 imp:n=1  $ top gap
C  === fuel universe 150 slot (4, 5, 1) ===
1550 1 -19.7147  -10        u=150 imp:n=1  $ right foil
1551 2 -19.4714  -11        u=150 imp:n=1  $ left foil
1552 3 -7.3896  -13        u=150 imp:n=1  $ bottom clad
1553 3 -7.3896  -14        u=150 imp:n=1  $ top clad
1554 0             -15 10 11  u=150 imp:n=1  $ foil-plane void
1555 0             -22 12     u=150 imp:n=1  $ void around clad
1556 4 -2.7338  -20        u=150 imp:n=1  $ 1in graphite
1557 4 -2.7338  -21        u=150 imp:n=1  $ 2in graphite
1558 4 -2.7338  -30        u=150 imp:n=1  $ left rail
1559 4 -2.7338  -31        u=150 imp:n=1  $ right rail
1560 0            -23        u=150 imp:n=1  $ top gap
C  === fuel universe 151 slot (4, 7, 1) ===
1561 1 -19.441  -10        u=151 imp:n=1  $ right foil
1562 2 -19.8212  -11        u=151 imp:n=1  $ left foil
1563 3 -7.3723  -13        u=151 imp:n=1  $ bottom clad
1564 3 -7.3723  -14        u=151 imp:n=1  $ top clad
1565 0             -15 10 11  u=151 imp:n=1  $ foil-plane void
1566 0             -22 12     u=151 imp:n=1  $ void around clad
1567 4 -2.7338  -20        u=151 imp:n=1  $ 1in graphite
1568 4 -2.7338  -21        u=151 imp:n=1  $ 2in graphite
1569 4 -2.7338  -30        u=151 imp:n=1  $ left rail
1570 4 -2.7338  -31        u=151 imp:n=1  $ right rail
1571 0            -23        u=151 imp:n=1  $ top gap
C  === fuel universe 152 slot (4, 2, 2) ===
1572 1 -18.1514  -10        u=152 imp:n=1  $ right foil
1573 2 -19.517  -11        u=152 imp:n=1  $ left foil
1574 3 -7.2707  -13        u=152 imp:n=1  $ bottom clad
1575 3 -7.2707  -14        u=152 imp:n=1  $ top clad
1576 0             -15 10 11  u=152 imp:n=1  $ foil-plane void
1577 0             -22 12     u=152 imp:n=1  $ void around clad
1578 4 -2.7338  -20        u=152 imp:n=1  $ 1in graphite
1579 4 -2.7338  -21        u=152 imp:n=1  $ 2in graphite
1580 4 -2.7338  -30        u=152 imp:n=1  $ left rail
1581 4 -2.7338  -31        u=152 imp:n=1  $ right rail
1582 0            -23        u=152 imp:n=1  $ top gap
C  === fuel universe 153 slot (4, 4, 2) ===
1583 1 -18.6076  -10        u=153 imp:n=1  $ right foil
1584 2 -19.3041  -11        u=153 imp:n=1  $ left foil
1585 3 -7.3337  -13        u=153 imp:n=1  $ bottom clad
1586 3 -7.3337  -14        u=153 imp:n=1  $ top clad
1587 0             -15 10 11  u=153 imp:n=1  $ foil-plane void
1588 0             -22 12     u=153 imp:n=1  $ void around clad
1589 4 -2.7338  -20        u=153 imp:n=1  $ 1in graphite
1590 4 -2.7338  -21        u=153 imp:n=1  $ 2in graphite
1591 4 -2.7338  -30        u=153 imp:n=1  $ left rail
1592 4 -2.7338  -31        u=153 imp:n=1  $ right rail
1593 0            -23        u=153 imp:n=1  $ top gap
C  === fuel universe 154 slot (4, 6, 2) ===
1594 1 -18.8966  -10        u=154 imp:n=1  $ right foil
1595 2 -18.7749  -11        u=154 imp:n=1  $ left foil
1596 3 -7.2707  -13        u=154 imp:n=1  $ bottom clad
1597 3 -7.2707  -14        u=154 imp:n=1  $ top clad
1598 0             -15 10 11  u=154 imp:n=1  $ foil-plane void
1599 0             -22 12     u=154 imp:n=1  $ void around clad
1600 4 -2.7338  -20        u=154 imp:n=1  $ 1in graphite
1601 4 -2.7338  -21        u=154 imp:n=1  $ 2in graphite
1602 4 -2.7338  -30        u=154 imp:n=1  $ left rail
1603 4 -2.7338  -31        u=154 imp:n=1  $ right rail
1604 0            -23        u=154 imp:n=1  $ top gap
C  === fuel universe 155 slot (4, 8, 2) ===
1605 1 -19.0912  -10        u=155 imp:n=1  $ right foil
1606 2 -19.0912  -11        u=155 imp:n=1  $ left foil
1607 3 -7.0655  -13        u=155 imp:n=1  $ bottom clad
1608 3 -7.0655  -14        u=155 imp:n=1  $ top clad
1609 0             -15 10 11  u=155 imp:n=1  $ foil-plane void
1610 0             -22 12     u=155 imp:n=1  $ void around clad
1611 4 -2.7338  -20        u=155 imp:n=1  $ 1in graphite
1612 4 -2.7338  -21        u=155 imp:n=1  $ 2in graphite
1613 4 -2.7338  -30        u=155 imp:n=1  $ left rail
1614 4 -2.7338  -31        u=155 imp:n=1  $ right rail
1615 0            -23        u=155 imp:n=1  $ top gap
C  === fuel universe 156 slot (4, 1, 3) ===
1616 1 -19.7755  -10        u=156 imp:n=1  $ right foil
1617 2 -19.6843  -11        u=156 imp:n=1  $ left foil
1618 3 -7.2707  -13        u=156 imp:n=1  $ bottom clad
1619 3 -7.2707  -14        u=156 imp:n=1  $ top clad
1620 0             -15 10 11  u=156 imp:n=1  $ foil-plane void
1621 0             -22 12     u=156 imp:n=1  $ void around clad
1622 4 -2.7338  -20        u=156 imp:n=1  $ 1in graphite
1623 4 -2.7338  -21        u=156 imp:n=1  $ 2in graphite
1624 4 -2.7338  -30        u=156 imp:n=1  $ left rail
1625 4 -2.7338  -31        u=156 imp:n=1  $ right rail
1626 0            -23        u=156 imp:n=1  $ top gap
C  === fuel universe 157 slot (4, 3, 3) ===
1627 1 -19.6387  -10        u=157 imp:n=1  $ right foil
1628 2 -19.6387  -11        u=157 imp:n=1  $ left foil
1629 3 -7.231  -13        u=157 imp:n=1  $ bottom clad
1630 3 -7.231  -14        u=157 imp:n=1  $ top clad
1631 0             -15 10 11  u=157 imp:n=1  $ foil-plane void
1632 0             -22 12     u=157 imp:n=1  $ void around clad
1633 4 -2.7338  -20        u=157 imp:n=1  $ 1in graphite
1634 4 -2.7338  -21        u=157 imp:n=1  $ 2in graphite
1635 4 -2.7338  -30        u=157 imp:n=1  $ left rail
1636 4 -2.7338  -31        u=157 imp:n=1  $ right rail
1637 0            -23        u=157 imp:n=1  $ top gap
C  === fuel universe 158 slot (4, 5, 3) ===
1638 1 -19.7147  -10        u=158 imp:n=1  $ right foil
1639 2 -19.6691  -11        u=158 imp:n=1  $ left foil
1640 3 -7.3448  -13        u=158 imp:n=1  $ bottom clad
1641 3 -7.3448  -14        u=158 imp:n=1  $ top clad
1642 0             -15 10 11  u=158 imp:n=1  $ foil-plane void
1643 0             -22 12     u=158 imp:n=1  $ void around clad
1644 4 -2.7338  -20        u=158 imp:n=1  $ 1in graphite
1645 4 -2.7338  -21        u=158 imp:n=1  $ 2in graphite
1646 4 -2.7338  -30        u=158 imp:n=1  $ left rail
1647 4 -2.7338  -31        u=158 imp:n=1  $ right rail
1648 0            -23        u=158 imp:n=1  $ top gap
C  === fuel universe 159 slot (4, 7, 3) ===
1649 1 -19.6843  -10        u=159 imp:n=1  $ right foil
1650 2 -19.6539  -11        u=159 imp:n=1  $ left foil
1651 3 -6.9414  -13        u=159 imp:n=1  $ bottom clad
1652 3 -6.9414  -14        u=159 imp:n=1  $ top clad
1653 0             -15 10 11  u=159 imp:n=1  $ foil-plane void
1654 0             -22 12     u=159 imp:n=1  $ void around clad
1655 4 -2.7338  -20        u=159 imp:n=1  $ 1in graphite
1656 4 -2.7338  -21        u=159 imp:n=1  $ 2in graphite
1657 4 -2.7338  -30        u=159 imp:n=1  $ left rail
1658 4 -2.7338  -31        u=159 imp:n=1  $ right rail
1659 0            -23        u=159 imp:n=1  $ top gap
C  === fuel universe 160 slot (4, 2, 4) ===
1660 1 -18.1514  -10        u=160 imp:n=1  $ right foil
1661 2 -19.517  -11        u=160 imp:n=1  $ left foil
1662 3 -7.2707  -13        u=160 imp:n=1  $ bottom clad
1663 3 -7.2707  -14        u=160 imp:n=1  $ top clad
1664 0             -15 10 11  u=160 imp:n=1  $ foil-plane void
1665 0             -22 12     u=160 imp:n=1  $ void around clad
1666 4 -2.7338  -20        u=160 imp:n=1  $ 1in graphite
1667 4 -2.7338  -21        u=160 imp:n=1  $ 2in graphite
1668 4 -2.7338  -30        u=160 imp:n=1  $ left rail
1669 4 -2.7338  -31        u=160 imp:n=1  $ right rail
1670 0            -23        u=160 imp:n=1  $ top gap
C  === fuel universe 161 slot (4, 4, 4) ===
1671 1 -18.6076  -10        u=161 imp:n=1  $ right foil
1672 2 -19.3041  -11        u=161 imp:n=1  $ left foil
1673 3 -7.3337  -13        u=161 imp:n=1  $ bottom clad
1674 3 -7.3337  -14        u=161 imp:n=1  $ top clad
1675 0             -15 10 11  u=161 imp:n=1  $ foil-plane void
1676 0             -22 12     u=161 imp:n=1  $ void around clad
1677 4 -2.7338  -20        u=161 imp:n=1  $ 1in graphite
1678 4 -2.7338  -21        u=161 imp:n=1  $ 2in graphite
1679 4 -2.7338  -30        u=161 imp:n=1  $ left rail
1680 4 -2.7338  -31        u=161 imp:n=1  $ right rail
1681 0            -23        u=161 imp:n=1  $ top gap
C  === fuel universe 162 slot (4, 6, 4) ===
1682 1 -18.8966  -10        u=162 imp:n=1  $ right foil
1683 2 -18.7749  -11        u=162 imp:n=1  $ left foil
1684 3 -7.2707  -13        u=162 imp:n=1  $ bottom clad
1685 3 -7.2707  -14        u=162 imp:n=1  $ top clad
1686 0             -15 10 11  u=162 imp:n=1  $ foil-plane void
1687 0             -22 12     u=162 imp:n=1  $ void around clad
1688 4 -2.7338  -20        u=162 imp:n=1  $ 1in graphite
1689 4 -2.7338  -21        u=162 imp:n=1  $ 2in graphite
1690 4 -2.7338  -30        u=162 imp:n=1  $ left rail
1691 4 -2.7338  -31        u=162 imp:n=1  $ right rail
1692 0            -23        u=162 imp:n=1  $ top gap
C  === fuel universe 163 slot (4, 8, 4) ===
1693 1 -19.0912  -10        u=163 imp:n=1  $ right foil
1694 2 -19.0912  -11        u=163 imp:n=1  $ left foil
1695 3 -7.0655  -13        u=163 imp:n=1  $ bottom clad
1696 3 -7.0655  -14        u=163 imp:n=1  $ top clad
1697 0             -15 10 11  u=163 imp:n=1  $ foil-plane void
1698 0             -22 12     u=163 imp:n=1  $ void around clad
1699 4 -2.7338  -20        u=163 imp:n=1  $ 1in graphite
1700 4 -2.7338  -21        u=163 imp:n=1  $ 2in graphite
1701 4 -2.7338  -30        u=163 imp:n=1  $ left rail
1702 4 -2.7338  -31        u=163 imp:n=1  $ right rail
1703 0            -23        u=163 imp:n=1  $ top gap
C  === fuel universe 164 slot (5, 2, 1) ===
1704 1 -19.2889  -10        u=164 imp:n=1  $ right foil
1705 2 -19.7907  -11        u=164 imp:n=1  $ left foil
1706 3 -7.2068  -13        u=164 imp:n=1  $ bottom clad
1707 3 -7.2068  -14        u=164 imp:n=1  $ top clad
1708 0             -15 10 11  u=164 imp:n=1  $ foil-plane void
1709 0             -22 12     u=164 imp:n=1  $ void around clad
1710 4 -2.7338  -20        u=164 imp:n=1  $ 1in graphite
1711 4 -2.7338  -21        u=164 imp:n=1  $ 2in graphite
1712 4 -2.7338  -30        u=164 imp:n=1  $ left rail
1713 4 -2.7338  -31        u=164 imp:n=1  $ right rail
1714 0            -23        u=164 imp:n=1  $ top gap
C  === fuel universe 165 slot (5, 4, 1) ===
1715 1 -18.6544  -10        u=165 imp:n=1  $ right foil
1716 2 -18.6072  -11        u=165 imp:n=1  $ left foil
1717 3 -7.2896  -13        u=165 imp:n=1  $ bottom clad
1718 3 -7.2896  -14        u=165 imp:n=1  $ top clad
1719 0             -15 10 11  u=165 imp:n=1  $ foil-plane void
1720 0             -22 12     u=165 imp:n=1  $ void around clad
1721 4 -2.7338  -20        u=165 imp:n=1  $ 1in graphite
1722 4 -2.7338  -21        u=165 imp:n=1  $ 2in graphite
1723 4 -2.7338  -30        u=165 imp:n=1  $ left rail
1724 4 -2.7338  -31        u=165 imp:n=1  $ right rail
1725 0            -23        u=165 imp:n=1  $ top gap
C  === fuel universe 166 slot (5, 6, 1) ===
1726 1 -19.5626  -10        u=166 imp:n=1  $ right foil
1727 2 -19.3193  -11        u=166 imp:n=1  $ left foil
1728 3 -7.0448  -13        u=166 imp:n=1  $ bottom clad
1729 3 -7.0448  -14        u=166 imp:n=1  $ top clad
1730 0             -15 10 11  u=166 imp:n=1  $ foil-plane void
1731 0             -22 12     u=166 imp:n=1  $ void around clad
1732 4 -2.7338  -20        u=166 imp:n=1  $ 1in graphite
1733 4 -2.7338  -21        u=166 imp:n=1  $ 2in graphite
1734 4 -2.7338  -30        u=166 imp:n=1  $ left rail
1735 4 -2.7338  -31        u=166 imp:n=1  $ right rail
1736 0            -23        u=166 imp:n=1  $ top gap
C  === fuel universe 167 slot (5, 8, 1) ===
1737 1 -19.9124  -10        u=167 imp:n=1  $ right foil
1738 2 -19.8972  -11        u=167 imp:n=1  $ left foil
1739 3 -7.0413  -13        u=167 imp:n=1  $ bottom clad
1740 3 -7.0413  -14        u=167 imp:n=1  $ top clad
1741 0             -15 10 11  u=167 imp:n=1  $ foil-plane void
1742 0             -22 12     u=167 imp:n=1  $ void around clad
1743 4 -2.7338  -20        u=167 imp:n=1  $ 1in graphite
1744 4 -2.7338  -21        u=167 imp:n=1  $ 2in graphite
1745 4 -2.7338  -30        u=167 imp:n=1  $ left rail
1746 4 -2.7338  -31        u=167 imp:n=1  $ right rail
1747 0            -23        u=167 imp:n=1  $ top gap
C  === fuel universe 168 slot (5, 1, 2) ===
1748 1 -17.2694  -10        u=168 imp:n=1  $ right foil
1749 2 -17.2846  -11        u=168 imp:n=1  $ left foil
1750 3 -7.4054  -13        u=168 imp:n=1  $ bottom clad
1751 3 -7.4054  -14        u=168 imp:n=1  $ top clad
1752 0             -15 10 11  u=168 imp:n=1  $ foil-plane void
1753 0             -22 12     u=168 imp:n=1  $ void around clad
1754 4 -2.7338  -20        u=168 imp:n=1  $ 1in graphite
1755 4 -2.7338  -21        u=168 imp:n=1  $ 2in graphite
1756 4 -2.7338  -30        u=168 imp:n=1  $ left rail
1757 4 -2.7338  -31        u=168 imp:n=1  $ right rail
1758 0            -23        u=168 imp:n=1  $ top gap
C  === fuel universe 169 slot (5, 3, 2) ===
1759 1 -18.7141  -10        u=169 imp:n=1  $ right foil
1760 2 -18.7597  -11        u=169 imp:n=1  $ left foil
1761 3 -7.2261  -13        u=169 imp:n=1  $ bottom clad
1762 3 -7.2261  -14        u=169 imp:n=1  $ top clad
1763 0             -15 10 11  u=169 imp:n=1  $ foil-plane void
1764 0             -22 12     u=169 imp:n=1  $ void around clad
1765 4 -2.7338  -20        u=169 imp:n=1  $ 1in graphite
1766 4 -2.7338  -21        u=169 imp:n=1  $ 2in graphite
1767 4 -2.7338  -30        u=169 imp:n=1  $ left rail
1768 4 -2.7338  -31        u=169 imp:n=1  $ right rail
1769 0            -23        u=169 imp:n=1  $ top gap
C  === fuel universe 170 slot (5, 5, 2) ===
1770 1 -19.7147  -10        u=170 imp:n=1  $ right foil
1771 2 -17.8777  -11        u=170 imp:n=1  $ left foil
1772 3 -7.4061  -13        u=170 imp:n=1  $ bottom clad
1773 3 -7.4061  -14        u=170 imp:n=1  $ top clad
1774 0             -15 10 11  u=170 imp:n=1  $ foil-plane void
1775 0             -22 12     u=170 imp:n=1  $ void around clad
1776 4 -2.7338  -20        u=170 imp:n=1  $ 1in graphite
1777 4 -2.7338  -21        u=170 imp:n=1  $ 2in graphite
1778 4 -2.7338  -30        u=170 imp:n=1  $ left rail
1779 4 -2.7338  -31        u=170 imp:n=1  $ right rail
1780 0            -23        u=170 imp:n=1  $ top gap
C  === fuel universe 171 slot (5, 7, 2) ===
1781 1 -18.7445  -10        u=171 imp:n=1  $ right foil
1782 2 -19.1368  -11        u=171 imp:n=1  $ left foil
1783 3 -7.2707  -13        u=171 imp:n=1  $ bottom clad
1784 3 -7.2707  -14        u=171 imp:n=1  $ top clad
1785 0             -15 10 11  u=171 imp:n=1  $ foil-plane void
1786 0             -22 12     u=171 imp:n=1  $ void around clad
1787 4 -2.7338  -20        u=171 imp:n=1  $ 1in graphite
1788 4 -2.7338  -21        u=171 imp:n=1  $ 2in graphite
1789 4 -2.7338  -30        u=171 imp:n=1  $ left rail
1790 4 -2.7338  -31        u=171 imp:n=1  $ right rail
1791 0            -23        u=171 imp:n=1  $ top gap
C  === fuel universe 172 slot (5, 2, 3) ===
1792 1 -17.2238  -10        u=172 imp:n=1  $ right foil
1793 2 -17.315  -11        u=172 imp:n=1  $ left foil
1794 3 -7.4089  -13        u=172 imp:n=1  $ bottom clad
1795 3 -7.4089  -14        u=172 imp:n=1  $ top clad
1796 0             -15 10 11  u=172 imp:n=1  $ foil-plane void
1797 0             -22 12     u=172 imp:n=1  $ void around clad
1798 4 -2.7338  -20        u=172 imp:n=1  $ 1in graphite
1799 4 -2.7338  -21        u=172 imp:n=1  $ 2in graphite
1800 4 -2.7338  -30        u=172 imp:n=1  $ left rail
1801 4 -2.7338  -31        u=172 imp:n=1  $ right rail
1802 0            -23        u=172 imp:n=1  $ top gap
C  === fuel universe 173 slot (5, 4, 3) ===
1803 1 -18.9574  -10        u=173 imp:n=1  $ right foil
1804 2 -19.7299  -11        u=173 imp:n=1  $ left foil
1805 3 -7.2707  -13        u=173 imp:n=1  $ bottom clad
1806 3 -7.2707  -14        u=173 imp:n=1  $ top clad
1807 0             -15 10 11  u=173 imp:n=1  $ foil-plane void
1808 0             -22 12     u=173 imp:n=1  $ void around clad
1809 4 -2.7338  -20        u=173 imp:n=1  $ 1in graphite
1810 4 -2.7338  -21        u=173 imp:n=1  $ 2in graphite
1811 4 -2.7338  -30        u=173 imp:n=1  $ left rail
1812 4 -2.7338  -31        u=173 imp:n=1  $ right rail
1813 0            -23        u=173 imp:n=1  $ top gap
C  === fuel universe 174 slot (5, 6, 3) ===
1814 1 -19.5931  -10        u=174 imp:n=1  $ right foil
1815 2 -19.4714  -11        u=174 imp:n=1  $ left foil
1816 3 -7.2792  -13        u=174 imp:n=1  $ bottom clad
1817 3 -7.2792  -14        u=174 imp:n=1  $ top clad
1818 0             -15 10 11  u=174 imp:n=1  $ foil-plane void
1819 0             -22 12     u=174 imp:n=1  $ void around clad
1820 4 -2.7338  -20        u=174 imp:n=1  $ 1in graphite
1821 4 -2.7338  -21        u=174 imp:n=1  $ 2in graphite
1822 4 -2.7338  -30        u=174 imp:n=1  $ left rail
1823 4 -2.7338  -31        u=174 imp:n=1  $ right rail
1824 0            -23        u=174 imp:n=1  $ top gap
C  === fuel universe 175 slot (5, 8, 3) ===
1825 1 -19.8212  -10        u=175 imp:n=1  $ right foil
1826 2 -19.958  -11        u=175 imp:n=1  $ left foil
1827 3 -7.2707  -13        u=175 imp:n=1  $ bottom clad
1828 3 -7.2707  -14        u=175 imp:n=1  $ top clad
1829 0             -15 10 11  u=175 imp:n=1  $ foil-plane void
1830 0             -22 12     u=175 imp:n=1  $ void around clad
1831 4 -2.7338  -20        u=175 imp:n=1  $ 1in graphite
1832 4 -2.7338  -21        u=175 imp:n=1  $ 2in graphite
1833 4 -2.7338  -30        u=175 imp:n=1  $ left rail
1834 4 -2.7338  -31        u=175 imp:n=1  $ right rail
1835 0            -23        u=175 imp:n=1  $ top gap
C  === fuel universe 176 slot (5, 1, 4) ===
1836 1 -17.2694  -10        u=176 imp:n=1  $ right foil
1837 2 -17.2846  -11        u=176 imp:n=1  $ left foil
1838 3 -7.4054  -13        u=176 imp:n=1  $ bottom clad
1839 3 -7.4054  -14        u=176 imp:n=1  $ top clad
1840 0             -15 10 11  u=176 imp:n=1  $ foil-plane void
1841 0             -22 12     u=176 imp:n=1  $ void around clad
1842 4 -2.7338  -20        u=176 imp:n=1  $ 1in graphite
1843 4 -2.7338  -21        u=176 imp:n=1  $ 2in graphite
1844 4 -2.7338  -30        u=176 imp:n=1  $ left rail
1845 4 -2.7338  -31        u=176 imp:n=1  $ right rail
1846 0            -23        u=176 imp:n=1  $ top gap
C  === fuel universe 177 slot (5, 3, 4) ===
1847 1 -18.7141  -10        u=177 imp:n=1  $ right foil
1848 2 -18.7597  -11        u=177 imp:n=1  $ left foil
1849 3 -7.2261  -13        u=177 imp:n=1  $ bottom clad
1850 3 -7.2261  -14        u=177 imp:n=1  $ top clad
1851 0             -15 10 11  u=177 imp:n=1  $ foil-plane void
1852 0             -22 12     u=177 imp:n=1  $ void around clad
1853 4 -2.7338  -20        u=177 imp:n=1  $ 1in graphite
1854 4 -2.7338  -21        u=177 imp:n=1  $ 2in graphite
1855 4 -2.7338  -30        u=177 imp:n=1  $ left rail
1856 4 -2.7338  -31        u=177 imp:n=1  $ right rail
1857 0            -23        u=177 imp:n=1  $ top gap
C  === fuel universe 178 slot (5, 5, 4) ===
1858 1 -19.7147  -10        u=178 imp:n=1  $ right foil
1859 2 -17.8777  -11        u=178 imp:n=1  $ left foil
1860 3 -7.4061  -13        u=178 imp:n=1  $ bottom clad
1861 3 -7.4061  -14        u=178 imp:n=1  $ top clad
1862 0             -15 10 11  u=178 imp:n=1  $ foil-plane void
1863 0             -22 12     u=178 imp:n=1  $ void around clad
1864 4 -2.7338  -20        u=178 imp:n=1  $ 1in graphite
1865 4 -2.7338  -21        u=178 imp:n=1  $ 2in graphite
1866 4 -2.7338  -30        u=178 imp:n=1  $ left rail
1867 4 -2.7338  -31        u=178 imp:n=1  $ right rail
1868 0            -23        u=178 imp:n=1  $ top gap
C  === fuel universe 179 slot (5, 7, 4) ===
1869 1 -18.7445  -10        u=179 imp:n=1  $ right foil
1870 2 -19.1368  -11        u=179 imp:n=1  $ left foil
1871 3 -7.2707  -13        u=179 imp:n=1  $ bottom clad
1872 3 -7.2707  -14        u=179 imp:n=1  $ top clad
1873 0             -15 10 11  u=179 imp:n=1  $ foil-plane void
1874 0             -22 12     u=179 imp:n=1  $ void around clad
1875 4 -2.7338  -20        u=179 imp:n=1  $ 1in graphite
1876 4 -2.7338  -21        u=179 imp:n=1  $ 2in graphite
1877 4 -2.7338  -30        u=179 imp:n=1  $ left rail
1878 4 -2.7338  -31        u=179 imp:n=1  $ right rail
1879 0            -23        u=179 imp:n=1  $ top gap
C  === fuel universe 180 slot (6, 1, 1) ===
1880 1 -17.8625  -10        u=180 imp:n=1  $ right foil
1881 2 -18.0602  -11        u=180 imp:n=1  $ left foil
1882 3 -7.2707  -13        u=180 imp:n=1  $ bottom clad
1883 3 -7.2707  -14        u=180 imp:n=1  $ top clad
1884 0             -15 10 11  u=180 imp:n=1  $ foil-plane void
1885 0             -22 12     u=180 imp:n=1  $ void around clad
1886 4 -2.7338  -20        u=180 imp:n=1  $ 1in graphite
1887 4 -2.7338  -21        u=180 imp:n=1  $ 2in graphite
1888 4 -2.7338  -30        u=180 imp:n=1  $ left rail
1889 4 -2.7338  -31        u=180 imp:n=1  $ right rail
1890 0            -23        u=180 imp:n=1  $ top gap
C  === fuel universe 181 slot (6, 3, 1) ===
1891 1 -17.8777  -10        u=181 imp:n=1  $ right foil
1892 2 -18.4403  -11        u=181 imp:n=1  $ left foil
1893 3 -7.1434  -13        u=181 imp:n=1  $ bottom clad
1894 3 -7.1434  -14        u=181 imp:n=1  $ top clad
1895 0             -15 10 11  u=181 imp:n=1  $ foil-plane void
1896 0             -22 12     u=181 imp:n=1  $ void around clad
1897 4 -2.7338  -20        u=181 imp:n=1  $ 1in graphite
1898 4 -2.7338  -21        u=181 imp:n=1  $ 2in graphite
1899 4 -2.7338  -30        u=181 imp:n=1  $ left rail
1900 4 -2.7338  -31        u=181 imp:n=1  $ right rail
1901 0            -23        u=181 imp:n=1  $ top gap
C  === fuel universe 182 slot (6, 5, 1) ===
1902 1 -18.1514  -10        u=182 imp:n=1  $ right foil
1903 2 -18.1058  -11        u=182 imp:n=1  $ left foil
1904 3 -7.2261  -13        u=182 imp:n=1  $ bottom clad
1905 3 -7.2261  -14        u=182 imp:n=1  $ top clad
1906 0             -15 10 11  u=182 imp:n=1  $ foil-plane void
1907 0             -22 12     u=182 imp:n=1  $ void around clad
1908 4 -2.7338  -20        u=182 imp:n=1  $ 1in graphite
1909 4 -2.7338  -21        u=182 imp:n=1  $ 2in graphite
1910 4 -2.7338  -30        u=182 imp:n=1  $ left rail
1911 4 -2.7338  -31        u=182 imp:n=1  $ right rail
1912 0            -23        u=182 imp:n=1  $ top gap
C  === fuel universe 183 slot (6, 7, 1) ===
1913 1 -18.2426  -10        u=183 imp:n=1  $ right foil
1914 2 -18.1058  -11        u=183 imp:n=1  $ left foil
1915 3 -7.2744  -13        u=183 imp:n=1  $ bottom clad
1916 3 -7.2744  -14        u=183 imp:n=1  $ top clad
1917 0             -15 10 11  u=183 imp:n=1  $ foil-plane void
1918 0             -22 12     u=183 imp:n=1  $ void around clad
1919 4 -2.7338  -20        u=183 imp:n=1  $ 1in graphite
1920 4 -2.7338  -21        u=183 imp:n=1  $ 2in graphite
1921 4 -2.7338  -30        u=183 imp:n=1  $ left rail
1922 4 -2.7338  -31        u=183 imp:n=1  $ right rail
1923 0            -23        u=183 imp:n=1  $ top gap
C  === fuel universe 184 slot (6, 2, 2) ===
1924 1 -18.5164  -10        u=184 imp:n=1  $ right foil
1925 2 -19.4866  -11        u=184 imp:n=1  $ left foil
1926 3 -7.1751  -13        u=184 imp:n=1  $ bottom clad
1927 3 -7.1751  -14        u=184 imp:n=1  $ top clad
1928 0             -15 10 11  u=184 imp:n=1  $ foil-plane void
1929 0             -22 12     u=184 imp:n=1  $ void around clad
1930 4 -2.7338  -20        u=184 imp:n=1  $ 1in graphite
1931 4 -2.7338  -21        u=184 imp:n=1  $ 2in graphite
1932 4 -2.7338  -30        u=184 imp:n=1  $ left rail
1933 4 -2.7338  -31        u=184 imp:n=1  $ right rail
1934 0            -23        u=184 imp:n=1  $ top gap
C  === fuel universe 185 slot (6, 4, 2) ===
1935 1 -19.2585  -10        u=185 imp:n=1  $ right foil
1936 2 -17.9993  -11        u=185 imp:n=1  $ left foil
1937 3 -7.0682  -13        u=185 imp:n=1  $ bottom clad
1938 3 -7.0682  -14        u=185 imp:n=1  $ top clad
1939 0             -15 10 11  u=185 imp:n=1  $ foil-plane void
1940 0             -22 12     u=185 imp:n=1  $ void around clad
1941 4 -2.7338  -20        u=185 imp:n=1  $ 1in graphite
1942 4 -2.7338  -21        u=185 imp:n=1  $ 2in graphite
1943 4 -2.7338  -30        u=185 imp:n=1  $ left rail
1944 4 -2.7338  -31        u=185 imp:n=1  $ right rail
1945 0            -23        u=185 imp:n=1  $ top gap
C  === fuel universe 186 slot (6, 6, 2) ===
1946 1 -18.121  -10        u=186 imp:n=1  $ right foil
1947 2 -18.9878  -11        u=186 imp:n=1  $ left foil
1948 3 -7.102  -13        u=186 imp:n=1  $ bottom clad
1949 3 -7.102  -14        u=186 imp:n=1  $ top clad
1950 0             -15 10 11  u=186 imp:n=1  $ foil-plane void
1951 0             -22 12     u=186 imp:n=1  $ void around clad
1952 4 -2.7338  -20        u=186 imp:n=1  $ 1in graphite
1953 4 -2.7338  -21        u=186 imp:n=1  $ 2in graphite
1954 4 -2.7338  -30        u=186 imp:n=1  $ left rail
1955 4 -2.7338  -31        u=186 imp:n=1  $ right rail
1956 0            -23        u=186 imp:n=1  $ top gap
C  === fuel universe 187 slot (6, 8, 2) ===
1957 1 -18.0145  -10        u=187 imp:n=1  $ right foil
1958 2 -18.2579  -11        u=187 imp:n=1  $ left foil
1959 3 -7.4985  -13        u=187 imp:n=1  $ bottom clad
1960 3 -7.4985  -14        u=187 imp:n=1  $ top clad
1961 0             -15 10 11  u=187 imp:n=1  $ foil-plane void
1962 0             -22 12     u=187 imp:n=1  $ void around clad
1963 4 -2.7338  -20        u=187 imp:n=1  $ 1in graphite
1964 4 -2.7338  -21        u=187 imp:n=1  $ 2in graphite
1965 4 -2.7338  -30        u=187 imp:n=1  $ left rail
1966 4 -2.7338  -31        u=187 imp:n=1  $ right rail
1967 0            -23        u=187 imp:n=1  $ top gap
C  === fuel universe 188 slot (6, 1, 3) ===
1968 1 -18.0297  -10        u=188 imp:n=1  $ right foil
1969 2 -17.7864  -11        u=188 imp:n=1  $ left foil
1970 3 -7.2707  -13        u=188 imp:n=1  $ bottom clad
1971 3 -7.2707  -14        u=188 imp:n=1  $ top clad
1972 0             -15 10 11  u=188 imp:n=1  $ foil-plane void
1973 0             -22 12     u=188 imp:n=1  $ void around clad
1974 4 -2.7338  -20        u=188 imp:n=1  $ 1in graphite
1975 4 -2.7338  -21        u=188 imp:n=1  $ 2in graphite
1976 4 -2.7338  -30        u=188 imp:n=1  $ left rail
1977 4 -2.7338  -31        u=188 imp:n=1  $ right rail
1978 0            -23        u=188 imp:n=1  $ top gap
C  === fuel universe 189 slot (6, 3, 3) ===
1979 1 -18.0602  -10        u=189 imp:n=1  $ right foil
1980 2 -17.9689  -11        u=189 imp:n=1  $ left foil
1981 3 -7.14  -13        u=189 imp:n=1  $ bottom clad
1982 3 -7.14  -14        u=189 imp:n=1  $ top clad
1983 0             -15 10 11  u=189 imp:n=1  $ foil-plane void
1984 0             -22 12     u=189 imp:n=1  $ void around clad
1985 4 -2.7338  -20        u=189 imp:n=1  $ 1in graphite
1986 4 -2.7338  -21        u=189 imp:n=1  $ 2in graphite
1987 4 -2.7338  -30        u=189 imp:n=1  $ left rail
1988 4 -2.7338  -31        u=189 imp:n=1  $ right rail
1989 0            -23        u=189 imp:n=1  $ top gap
C  === fuel universe 190 slot (6, 5, 3) ===
1990 1 -18.0602  -10        u=190 imp:n=1  $ right foil
1991 2 -17.8777  -11        u=190 imp:n=1  $ left foil
1992 3 -7.2707  -13        u=190 imp:n=1  $ bottom clad
1993 3 -7.2707  -14        u=190 imp:n=1  $ top clad
1994 0             -15 10 11  u=190 imp:n=1  $ foil-plane void
1995 0             -22 12     u=190 imp:n=1  $ void around clad
1996 4 -2.7338  -20        u=190 imp:n=1  $ 1in graphite
1997 4 -2.7338  -21        u=190 imp:n=1  $ 2in graphite
1998 4 -2.7338  -30        u=190 imp:n=1  $ left rail
1999 4 -2.7338  -31        u=190 imp:n=1  $ right rail
2000 0            -23        u=190 imp:n=1  $ top gap
C  === fuel universe 191 slot (6, 7, 3) ===
2001 1 -17.9081  -10        u=191 imp:n=1  $ right foil
2002 2 -17.8929  -11        u=191 imp:n=1  $ left foil
2003 3 -7.2707  -13        u=191 imp:n=1  $ bottom clad
2004 3 -7.2707  -14        u=191 imp:n=1  $ top clad
2005 0             -15 10 11  u=191 imp:n=1  $ foil-plane void
2006 0             -22 12     u=191 imp:n=1  $ void around clad
2007 4 -2.7338  -20        u=191 imp:n=1  $ 1in graphite
2008 4 -2.7338  -21        u=191 imp:n=1  $ 2in graphite
2009 4 -2.7338  -30        u=191 imp:n=1  $ left rail
2010 4 -2.7338  -31        u=191 imp:n=1  $ right rail
2011 0            -23        u=191 imp:n=1  $ top gap
C  === fuel universe 192 slot (6, 2, 4) ===
2012 1 -18.5164  -10        u=192 imp:n=1  $ right foil
2013 2 -19.4866  -11        u=192 imp:n=1  $ left foil
2014 3 -7.1751  -13        u=192 imp:n=1  $ bottom clad
2015 3 -7.1751  -14        u=192 imp:n=1  $ top clad
2016 0             -15 10 11  u=192 imp:n=1  $ foil-plane void
2017 0             -22 12     u=192 imp:n=1  $ void around clad
2018 4 -2.7338  -20        u=192 imp:n=1  $ 1in graphite
2019 4 -2.7338  -21        u=192 imp:n=1  $ 2in graphite
2020 4 -2.7338  -30        u=192 imp:n=1  $ left rail
2021 4 -2.7338  -31        u=192 imp:n=1  $ right rail
2022 0            -23        u=192 imp:n=1  $ top gap
C  === fuel universe 193 slot (6, 4, 4) ===
2023 1 -19.2585  -10        u=193 imp:n=1  $ right foil
2024 2 -17.9993  -11        u=193 imp:n=1  $ left foil
2025 3 -7.0682  -13        u=193 imp:n=1  $ bottom clad
2026 3 -7.0682  -14        u=193 imp:n=1  $ top clad
2027 0             -15 10 11  u=193 imp:n=1  $ foil-plane void
2028 0             -22 12     u=193 imp:n=1  $ void around clad
2029 4 -2.7338  -20        u=193 imp:n=1  $ 1in graphite
2030 4 -2.7338  -21        u=193 imp:n=1  $ 2in graphite
2031 4 -2.7338  -30        u=193 imp:n=1  $ left rail
2032 4 -2.7338  -31        u=193 imp:n=1  $ right rail
2033 0            -23        u=193 imp:n=1  $ top gap
C  === fuel universe 194 slot (6, 6, 4) ===
2034 1 -18.121  -10        u=194 imp:n=1  $ right foil
2035 2 -18.9878  -11        u=194 imp:n=1  $ left foil
2036 3 -7.102  -13        u=194 imp:n=1  $ bottom clad
2037 3 -7.102  -14        u=194 imp:n=1  $ top clad
2038 0             -15 10 11  u=194 imp:n=1  $ foil-plane void
2039 0             -22 12     u=194 imp:n=1  $ void around clad
2040 4 -2.7338  -20        u=194 imp:n=1  $ 1in graphite
2041 4 -2.7338  -21        u=194 imp:n=1  $ 2in graphite
2042 4 -2.7338  -30        u=194 imp:n=1  $ left rail
2043 4 -2.7338  -31        u=194 imp:n=1  $ right rail
2044 0            -23        u=194 imp:n=1  $ top gap
C  === fuel universe 195 slot (6, 8, 4) ===
2045 1 -18.0145  -10        u=195 imp:n=1  $ right foil
2046 2 -18.2579  -11        u=195 imp:n=1  $ left foil
2047 3 -7.4985  -13        u=195 imp:n=1  $ bottom clad
2048 3 -7.4985  -14        u=195 imp:n=1  $ top clad
2049 0             -15 10 11  u=195 imp:n=1  $ foil-plane void
2050 0             -22 12     u=195 imp:n=1  $ void around clad
2051 4 -2.7338  -20        u=195 imp:n=1  $ 1in graphite
2052 4 -2.7338  -21        u=195 imp:n=1  $ 2in graphite
2053 4 -2.7338  -30        u=195 imp:n=1  $ left rail
2054 4 -2.7338  -31        u=195 imp:n=1  $ right rail
2055 0            -23        u=195 imp:n=1  $ top gap

C  Surface cards
C  --- fuel element (local coords, foil plane at z = 0) ---
10 rpp   0.23749  11.35126  -29.1211  29.1211  -0.00254  0.00254  $ right U foil
11 rpp -11.35126  -0.23749  -29.1211  29.1211  -0.00254  0.00254  $ left U foil
12 rpp -11.7475   11.7475  -30.45968 30.45968  -0.00762  0.00762  $ clad envelope
13 rpp -11.7475   11.7475  -30.45968 30.45968  -0.00762 -0.00254  $ bottom clad sheet
14 rpp -11.7475   11.7475  -30.45968 30.45968   0.00254  0.00762  $ top clad sheet
15 rpp -11.7475   11.7475  -30.45968 30.45968  -0.00254  0.00254  $ foil mid-plane
C  --- graphite ---
20 rpp -12.065    12.065   -30.45968 30.45968  -2.44348 -0.00762  $ 1in block (below)
21 rpp -12.065    12.065   -30.45968 30.45968   0.09652  5.08762  $ 2in block (above)
22 rpp -12.065    12.065   -30.45968 30.45968  -0.00762  0.09652  $ mid slab footprint
23 rpp -12.065    12.065   -30.45968 30.45968   5.08762  5.17652  $ top gap
30 rpp -15.22984 -12.065   -30.45968 30.45968  -2.44348  5.17652  $ left rail
31 rpp  12.065    15.22984 -30.45968 30.45968  -2.44348  5.17652  $ right rail
C  --- lattice element, container, world ---
99 rpp -15.22984  15.22984 -30.45968 30.45968  -2.44348  5.17652  $ lattice element
98 rpp -15.22984 228.44760 -30.45968 213.21776 -2.44348 43.27652  $ lattice container
999 rpp -500 500 -500 600 -500 600  $ world

C  Data cards
mode n
kcode 10000 1.0 30 130
ksrc 24.90343 0.00000 0.00000
     36.01593 0.00000 0.00000
     85.82279 0.00000 0.00000
     96.93529 0.00000 0.00000
     146.74215 0.00000 0.00000
     157.85465 0.00000 0.00000
     207.66151 0.00000 0.00000
     218.77401 0.00000 0.00000
     -5.55625 60.91936 0.00000
     5.55625 60.91936 0.00000
     55.36311 60.91936 0.00000
     66.47561 60.91936 0.00000
     116.28247 60.91936 0.00000
     127.39497 60.91936 0.00000
     177.20183 60.91936 0.00000
     188.31433 60.91936 0.00000
     24.90343 121.83872 0.00000
     36.01593 121.83872 0.00000
     85.82279 121.83872 0.00000
     96.93529 121.83872 0.00000
     146.74215 121.83872 0.00000
     157.85465 121.83872 0.00000
     207.66151 121.83872 0.00000
     218.77401 121.83872 0.00000
     -5.55625 182.75808 0.00000
     5.55625 182.75808 0.00000
     55.36311 182.75808 0.00000
     66.47561 182.75808 0.00000
     116.28247 182.75808 0.00000
     127.39497 182.75808 0.00000
     177.20183 182.75808 0.00000
     188.31433 182.75808 0.00000
     -5.55625 0.00000 7.62000
     5.55625 0.00000 7.62000
     55.36311 0.00000 7.62000
     66.47561 0.00000 7.62000
     116.28247 0.00000 7.62000
     127.39497 0.00000 7.62000
     177.20183 0.00000 7.62000
     188.31433 0.00000 7.62000
     24.90343 60.91936 7.62000
     36.01593 60.91936 7.62000
     85.82279 60.91936 7.62000
     96.93529 60.91936 7.62000
     146.74215 60.91936 7.62000
     157.85465 60.91936 7.62000
     207.66151 60.91936 7.62000
     218.77401 60.91936 7.62000
     -5.55625 121.83872 7.62000
     5.55625 121.83872 7.62000
     55.36311 121.83872 7.62000
     66.47561 121.83872 7.62000
     116.28247 121.83872 7.62000
     127.39497 121.83872 7.62000
     177.20183 121.83872 7.62000
     188.31433 121.83872 7.62000
     24.90343 182.75808 7.62000
     36.01593 182.75808 7.62000
     85.82279 182.75808 7.62000
     96.93529 182.75808 7.62000
     146.74215 182.75808 7.62000
     157.85465 182.75808 7.62000
     207.66151 182.75808 7.62000
     218.77401 182.75808 7.62000
     24.90343 0.00000 15.24000
     36.01593 0.00000 15.24000
     85.82279 0.00000 15.24000
     96.93529 0.00000 15.24000
     146.74215 0.00000 15.24000
     157.85465 0.00000 15.24000
     207.66151 0.00000 15.24000
     218.77401 0.00000 15.24000
     -5.55625 60.91936 15.24000
     5.55625 60.91936 15.24000
     55.36311 60.91936 15.24000
     66.47561 60.91936 15.24000
     116.28247 60.91936 15.24000
     127.39497 60.91936 15.24000
     177.20183 60.91936 15.24000
     188.31433 60.91936 15.24000
     24.90343 121.83872 15.24000
     36.01593 121.83872 15.24000
     85.82279 121.83872 15.24000
     96.93529 121.83872 15.24000
     146.74215 121.83872 15.24000
     157.85465 121.83872 15.24000
     207.66151 121.83872 15.24000
     218.77401 121.83872 15.24000
     -5.55625 182.75808 15.24000
     5.55625 182.75808 15.24000
     55.36311 182.75808 15.24000
     66.47561 182.75808 15.24000
     116.28247 182.75808 15.24000
     127.39497 182.75808 15.24000
     177.20183 182.75808 15.24000
     188.31433 182.75808 15.24000
     -5.55625 0.00000 22.86000
     5.55625 0.00000 22.86000
     55.36311 0.00000 22.86000
     66.47561 0.00000 22.86000
     116.28247 0.00000 22.86000
     127.39497 0.00000 22.86000
     177.20183 0.00000 22.86000
     188.31433 0.00000 22.86000
     24.90343 60.91936 22.86000
     36.01593 60.91936 22.86000
     85.82279 60.91936 22.86000
     96.93529 60.91936 22.86000
     146.74215 60.91936 22.86000
     157.85465 60.91936 22.86000
     207.66151 60.91936 22.86000
     218.77401 60.91936 22.86000
     -5.55625 121.83872 22.86000
     5.55625 121.83872 22.86000
     55.36311 121.83872 22.86000
     66.47561 121.83872 22.86000
     116.28247 121.83872 22.86000
     127.39497 121.83872 22.86000
     177.20183 121.83872 22.86000
     188.31433 121.83872 22.86000
     24.90343 182.75808 22.86000
     36.01593 182.75808 22.86000
     85.82279 182.75808 22.86000
     96.93529 182.75808 22.86000
     146.74215 182.75808 22.86000
     157.85465 182.75808 22.86000
     207.66151 182.75808 22.86000
     218.77401 182.75808 22.86000
     24.90343 0.00000 30.48000
     36.01593 0.00000 30.48000
     85.82279 0.00000 30.48000
     96.93529 0.00000 30.48000
     146.74215 0.00000 30.48000
     157.85465 0.00000 30.48000
     207.66151 0.00000 30.48000
     218.77401 0.00000 30.48000
     -5.55625 60.91936 30.48000
     5.55625 60.91936 30.48000
     55.36311 60.91936 30.48000
     66.47561 60.91936 30.48000
     116.28247 60.91936 30.48000
     127.39497 60.91936 30.48000
     177.20183 60.91936 30.48000
     188.31433 60.91936 30.48000
     24.90343 121.83872 30.48000
     36.01593 121.83872 30.48000
     85.82279 121.83872 30.48000
     96.93529 121.83872 30.48000
     146.74215 121.83872 30.48000
     157.85465 121.83872 30.48000
     207.66151 121.83872 30.48000
     218.77401 121.83872 30.48000
     -5.55625 182.75808 30.48000
     5.55625 182.75808 30.48000
     55.36311 182.75808 30.48000
     66.47561 182.75808 30.48000
     116.28247 182.75808 30.48000
     127.39497 182.75808 30.48000
     177.20183 182.75808 30.48000
     188.31433 182.75808 30.48000
     -5.55625 0.00000 38.10000
     5.55625 0.00000 38.10000
     55.36311 0.00000 38.10000
     66.47561 0.00000 38.10000
     116.28247 0.00000 38.10000
     127.39497 0.00000 38.10000
     177.20183 0.00000 38.10000
     188.31433 0.00000 38.10000
     24.90343 60.91936 38.10000
     36.01593 60.91936 38.10000
     85.82279 60.91936 38.10000
     96.93529 60.91936 38.10000
     146.74215 60.91936 38.10000
     157.85465 60.91936 38.10000
     207.66151 60.91936 38.10000
     218.77401 60.91936 38.10000
     -5.55625 121.83872 38.10000
     5.55625 121.83872 38.10000
     55.36311 121.83872 38.10000
     66.47561 121.83872 38.10000
     116.28247 121.83872 38.10000
     127.39497 121.83872 38.10000
     177.20183 121.83872 38.10000
     188.31433 121.83872 38.10000
     24.90343 182.75808 38.10000
     36.01593 182.75808 38.10000
     85.82279 182.75808 38.10000
     96.93529 182.75808 38.10000
     146.74215 182.75808 38.10000
     157.85465 182.75808 38.10000
     207.66151 182.75808 38.10000
     218.77401 182.75808 38.10000
C  Materials cards
m1   92234.80c 5.93E-04   92235.80c 4.57E-02   $ U for right foil
     92236.80c 3.43E-03   92238.80c 2.94E-03
m2   92234.80c 5.87E-04   92235.80c 4.53E-02   $ U for left foil
     92236.80c 3.40E-03   92238.80c 2.91E-03
m3   6000.80c  3.11E-03   25055.80c 1.70E-03   $ stainless-steel cladding
     14028.80c 1.54E-03   14029.80c 7.54E-05   14030.80c 4.81E-05
     24050.80c 7.31E-04   24052.80c 1.35E-02   24053.80c 1.51E-03
     24054.80c 3.68E-04   28058.80c 6.03E-03   28060.80c 2.25E-03
     28061.80c 9.60E-05   28062.80c 3.01E-04   28064.80c 7.43E-05
     15031.80c 6.78E-04   16032.80c 4.16E-04   16033.80c 3.18E-06
     16034.80c 1.75E-05   16036.80c 3.89E-08   26054.80c 3.36E-03
     26056.80c 5.09E-02   26057.80c 1.15E-03   26058.80c 1.51E-04
m4   6000.80c  1   $ graphite
mt4  grph.20t
