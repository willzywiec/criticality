Hot Box 8x4x11 (measured density)
C  Cell cards
1 0 -99 lat=1 u=1 imp:n=1
      fill=0:7 0:3 0:10
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
         2  196    2  197    2  198    2  199  $ row j=0 layer k=6
       200    2  201    2  202    2  203    2  $ row j=1 layer k=6
         2  204    2  205    2  206    2  207  $ row j=2 layer k=6
       208    2  209    2  210    2  211    2  $ row j=3 layer k=6
       212    2  213    2  214    2  215    2  $ row j=0 layer k=7
         2  216    2  217    2  218    2  219  $ row j=1 layer k=7
       220    2  221    2  222    2  223    2  $ row j=2 layer k=7
         2  224    2  225    2  226    2  227  $ row j=3 layer k=7
         2  228    2  229    2  230    2  231  $ row j=0 layer k=8
       232    2  233    2  234    2  235    2  $ row j=1 layer k=8
         2  236    2  237    2  238    2  239  $ row j=2 layer k=8
       240    2  241    2  242    2  243    2  $ row j=3 layer k=8
       244    2  245    2  246    2  247    2  $ row j=0 layer k=9
         2  248    2  249    2  250    2  251  $ row j=1 layer k=9
       252    2  253    2  254    2  255    2  $ row j=2 layer k=9
         2  256    2  257    2  258    2  259  $ row j=3 layer k=9
         2  260    2  261    2  262    2  263  $ row j=0 layer k=10
       264    2  265    2  266    2  267    2  $ row j=1 layer k=10
         2  268    2  269    2  270    2  271  $ row j=2 layer k=10
       272    2  273    2  274    2  275    2  $ row j=3 layer k=10
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
C  === fuel universe 196 slot (7, 2, 1) ===
2056 1 -17.9537  -10        u=196 imp:n=1  $ right foil
2057 2 -18.2426  -11        u=196 imp:n=1  $ left foil
2058 3 -7.2399  -13        u=196 imp:n=1  $ bottom clad
2059 3 -7.2399  -14        u=196 imp:n=1  $ top clad
2060 0             -15 10 11  u=196 imp:n=1  $ foil-plane void
2061 0             -22 12     u=196 imp:n=1  $ void around clad
2062 4 -2.7338  -20        u=196 imp:n=1  $ 1in graphite
2063 4 -2.7338  -21        u=196 imp:n=1  $ 2in graphite
2064 4 -2.7338  -30        u=196 imp:n=1  $ left rail
2065 4 -2.7338  -31        u=196 imp:n=1  $ right rail
2066 0            -23        u=196 imp:n=1  $ top gap
C  === fuel universe 197 slot (7, 4, 1) ===
2067 1 -17.9537  -10        u=197 imp:n=1  $ right foil
2068 2 -18.3947  -11        u=197 imp:n=1  $ left foil
2069 3 -7.2055  -13        u=197 imp:n=1  $ bottom clad
2070 3 -7.2055  -14        u=197 imp:n=1  $ top clad
2071 0             -15 10 11  u=197 imp:n=1  $ foil-plane void
2072 0             -22 12     u=197 imp:n=1  $ void around clad
2073 4 -2.7338  -20        u=197 imp:n=1  $ 1in graphite
2074 4 -2.7338  -21        u=197 imp:n=1  $ 2in graphite
2075 4 -2.7338  -30        u=197 imp:n=1  $ left rail
2076 4 -2.7338  -31        u=197 imp:n=1  $ right rail
2077 0            -23        u=197 imp:n=1  $ top gap
C  === fuel universe 198 slot (7, 6, 1) ===
2078 1 -18.1666  -10        u=198 imp:n=1  $ right foil
2079 2 -18.2122  -11        u=198 imp:n=1  $ left foil
2080 3 -7.2675  -13        u=198 imp:n=1  $ bottom clad
2081 3 -7.2675  -14        u=198 imp:n=1  $ top clad
2082 0             -15 10 11  u=198 imp:n=1  $ foil-plane void
2083 0             -22 12     u=198 imp:n=1  $ void around clad
2084 4 -2.7338  -20        u=198 imp:n=1  $ 1in graphite
2085 4 -2.7338  -21        u=198 imp:n=1  $ 2in graphite
2086 4 -2.7338  -30        u=198 imp:n=1  $ left rail
2087 4 -2.7338  -31        u=198 imp:n=1  $ right rail
2088 0            -23        u=198 imp:n=1  $ top gap
C  === fuel universe 199 slot (7, 8, 1) ===
2089 1 -18.0145  -10        u=199 imp:n=1  $ right foil
2090 2 -18.0754  -11        u=199 imp:n=1  $ left foil
2091 3 -7.2707  -13        u=199 imp:n=1  $ bottom clad
2092 3 -7.2707  -14        u=199 imp:n=1  $ top clad
2093 0             -15 10 11  u=199 imp:n=1  $ foil-plane void
2094 0             -22 12     u=199 imp:n=1  $ void around clad
2095 4 -2.7338  -20        u=199 imp:n=1  $ 1in graphite
2096 4 -2.7338  -21        u=199 imp:n=1  $ 2in graphite
2097 4 -2.7338  -30        u=199 imp:n=1  $ left rail
2098 4 -2.7338  -31        u=199 imp:n=1  $ right rail
2099 0            -23        u=199 imp:n=1  $ top gap
C  === fuel universe 200 slot (7, 1, 2) ===
2100 1 -18.3613  -10        u=200 imp:n=1  $ right foil
2101 2 -18.0906  -11        u=200 imp:n=1  $ left foil
2102 3 -7.251  -13        u=200 imp:n=1  $ bottom clad
2103 3 -7.251  -14        u=200 imp:n=1  $ top clad
2104 0             -15 10 11  u=200 imp:n=1  $ foil-plane void
2105 0             -22 12     u=200 imp:n=1  $ void around clad
2106 4 -2.7338  -20        u=200 imp:n=1  $ 1in graphite
2107 4 -2.7338  -21        u=200 imp:n=1  $ 2in graphite
2108 4 -2.7338  -30        u=200 imp:n=1  $ left rail
2109 4 -2.7338  -31        u=200 imp:n=1  $ right rail
2110 0            -23        u=200 imp:n=1  $ top gap
C  === fuel universe 201 slot (7, 3, 2) ===
2111 1 -18.486  -10        u=201 imp:n=1  $ right foil
2112 2 -18.6228  -11        u=201 imp:n=1  $ left foil
2113 3 -7.7226  -13        u=201 imp:n=1  $ bottom clad
2114 3 -7.7226  -14        u=201 imp:n=1  $ top clad
2115 0             -15 10 11  u=201 imp:n=1  $ foil-plane void
2116 0             -22 12     u=201 imp:n=1  $ void around clad
2117 4 -2.7338  -20        u=201 imp:n=1  $ 1in graphite
2118 4 -2.7338  -21        u=201 imp:n=1  $ 2in graphite
2119 4 -2.7338  -30        u=201 imp:n=1  $ left rail
2120 4 -2.7338  -31        u=201 imp:n=1  $ right rail
2121 0            -23        u=201 imp:n=1  $ top gap
C  === fuel universe 202 slot (7, 5, 2) ===
2122 1 -17.7864  -10        u=202 imp:n=1  $ right foil
2123 2 -19.3345  -11        u=202 imp:n=1  $ left foil
2124 3 -7.2707  -13        u=202 imp:n=1  $ bottom clad
2125 3 -7.2707  -14        u=202 imp:n=1  $ top clad
2126 0             -15 10 11  u=202 imp:n=1  $ foil-plane void
2127 0             -22 12     u=202 imp:n=1  $ void around clad
2128 4 -2.7338  -20        u=202 imp:n=1  $ 1in graphite
2129 4 -2.7338  -21        u=202 imp:n=1  $ 2in graphite
2130 4 -2.7338  -30        u=202 imp:n=1  $ left rail
2131 4 -2.7338  -31        u=202 imp:n=1  $ right rail
2132 0            -23        u=202 imp:n=1  $ top gap
C  === fuel universe 203 slot (7, 7, 2) ===
2133 1 -18.6837  -10        u=203 imp:n=1  $ right foil
2134 2 -18.2731  -11        u=203 imp:n=1  $ left foil
2135 3 -7.4123  -13        u=203 imp:n=1  $ bottom clad
2136 3 -7.4123  -14        u=203 imp:n=1  $ top clad
2137 0             -15 10 11  u=203 imp:n=1  $ foil-plane void
2138 0             -22 12     u=203 imp:n=1  $ void around clad
2139 4 -2.7338  -20        u=203 imp:n=1  $ 1in graphite
2140 4 -2.7338  -21        u=203 imp:n=1  $ 2in graphite
2141 4 -2.7338  -30        u=203 imp:n=1  $ left rail
2142 4 -2.7338  -31        u=203 imp:n=1  $ right rail
2143 0            -23        u=203 imp:n=1  $ top gap
C  === fuel universe 204 slot (7, 2, 3) ===
2144 1 -17.9385  -10        u=204 imp:n=1  $ right foil
2145 2 -18.4251  -11        u=204 imp:n=1  $ left foil
2146 3 -7.4089  -13        u=204 imp:n=1  $ bottom clad
2147 3 -7.4089  -14        u=204 imp:n=1  $ top clad
2148 0             -15 10 11  u=204 imp:n=1  $ foil-plane void
2149 0             -22 12     u=204 imp:n=1  $ void around clad
2150 4 -2.7338  -20        u=204 imp:n=1  $ 1in graphite
2151 4 -2.7338  -21        u=204 imp:n=1  $ 2in graphite
2152 4 -2.7338  -30        u=204 imp:n=1  $ left rail
2153 4 -2.7338  -31        u=204 imp:n=1  $ right rail
2154 0            -23        u=204 imp:n=1  $ top gap
C  === fuel universe 205 slot (7, 4, 3) ===
2155 1 -18.0602  -10        u=205 imp:n=1  $ right foil
2156 2 -18.0297  -11        u=205 imp:n=1  $ left foil
2157 3 -7.2707  -13        u=205 imp:n=1  $ bottom clad
2158 3 -7.2707  -14        u=205 imp:n=1  $ top clad
2159 0             -15 10 11  u=205 imp:n=1  $ foil-plane void
2160 0             -22 12     u=205 imp:n=1  $ void around clad
2161 4 -2.7338  -20        u=205 imp:n=1  $ 1in graphite
2162 4 -2.7338  -21        u=205 imp:n=1  $ 2in graphite
2163 4 -2.7338  -30        u=205 imp:n=1  $ left rail
2164 4 -2.7338  -31        u=205 imp:n=1  $ right rail
2165 0            -23        u=205 imp:n=1  $ top gap
C  === fuel universe 206 slot (7, 6, 3) ===
2166 1 -18.0602  -10        u=206 imp:n=1  $ right foil
2167 2 -17.9841  -11        u=206 imp:n=1  $ left foil
2168 3 -7.2707  -13        u=206 imp:n=1  $ bottom clad
2169 3 -7.2707  -14        u=206 imp:n=1  $ top clad
2170 0             -15 10 11  u=206 imp:n=1  $ foil-plane void
2171 0             -22 12     u=206 imp:n=1  $ void around clad
2172 4 -2.7338  -20        u=206 imp:n=1  $ 1in graphite
2173 4 -2.7338  -21        u=206 imp:n=1  $ 2in graphite
2174 4 -2.7338  -30        u=206 imp:n=1  $ left rail
2175 4 -2.7338  -31        u=206 imp:n=1  $ right rail
2176 0            -23        u=206 imp:n=1  $ top gap
C  === fuel universe 207 slot (7, 8, 3) ===
2177 1 -18.0297  -10        u=207 imp:n=1  $ right foil
2178 2 -17.9081  -11        u=207 imp:n=1  $ left foil
2179 3 -7.2707  -13        u=207 imp:n=1  $ bottom clad
2180 3 -7.2707  -14        u=207 imp:n=1  $ top clad
2181 0             -15 10 11  u=207 imp:n=1  $ foil-plane void
2182 0             -22 12     u=207 imp:n=1  $ void around clad
2183 4 -2.7338  -20        u=207 imp:n=1  $ 1in graphite
2184 4 -2.7338  -21        u=207 imp:n=1  $ 2in graphite
2185 4 -2.7338  -30        u=207 imp:n=1  $ left rail
2186 4 -2.7338  -31        u=207 imp:n=1  $ right rail
2187 0            -23        u=207 imp:n=1  $ top gap
C  === fuel universe 208 slot (7, 1, 4) ===
2188 1 -18.3613  -10        u=208 imp:n=1  $ right foil
2189 2 -18.0906  -11        u=208 imp:n=1  $ left foil
2190 3 -7.251  -13        u=208 imp:n=1  $ bottom clad
2191 3 -7.251  -14        u=208 imp:n=1  $ top clad
2192 0             -15 10 11  u=208 imp:n=1  $ foil-plane void
2193 0             -22 12     u=208 imp:n=1  $ void around clad
2194 4 -2.7338  -20        u=208 imp:n=1  $ 1in graphite
2195 4 -2.7338  -21        u=208 imp:n=1  $ 2in graphite
2196 4 -2.7338  -30        u=208 imp:n=1  $ left rail
2197 4 -2.7338  -31        u=208 imp:n=1  $ right rail
2198 0            -23        u=208 imp:n=1  $ top gap
C  === fuel universe 209 slot (7, 3, 4) ===
2199 1 -18.486  -10        u=209 imp:n=1  $ right foil
2200 2 -18.6228  -11        u=209 imp:n=1  $ left foil
2201 3 -7.7226  -13        u=209 imp:n=1  $ bottom clad
2202 3 -7.7226  -14        u=209 imp:n=1  $ top clad
2203 0             -15 10 11  u=209 imp:n=1  $ foil-plane void
2204 0             -22 12     u=209 imp:n=1  $ void around clad
2205 4 -2.7338  -20        u=209 imp:n=1  $ 1in graphite
2206 4 -2.7338  -21        u=209 imp:n=1  $ 2in graphite
2207 4 -2.7338  -30        u=209 imp:n=1  $ left rail
2208 4 -2.7338  -31        u=209 imp:n=1  $ right rail
2209 0            -23        u=209 imp:n=1  $ top gap
C  === fuel universe 210 slot (7, 5, 4) ===
2210 1 -17.7864  -10        u=210 imp:n=1  $ right foil
2211 2 -19.3345  -11        u=210 imp:n=1  $ left foil
2212 3 -7.2707  -13        u=210 imp:n=1  $ bottom clad
2213 3 -7.2707  -14        u=210 imp:n=1  $ top clad
2214 0             -15 10 11  u=210 imp:n=1  $ foil-plane void
2215 0             -22 12     u=210 imp:n=1  $ void around clad
2216 4 -2.7338  -20        u=210 imp:n=1  $ 1in graphite
2217 4 -2.7338  -21        u=210 imp:n=1  $ 2in graphite
2218 4 -2.7338  -30        u=210 imp:n=1  $ left rail
2219 4 -2.7338  -31        u=210 imp:n=1  $ right rail
2220 0            -23        u=210 imp:n=1  $ top gap
C  === fuel universe 211 slot (7, 7, 4) ===
2221 1 -18.6837  -10        u=211 imp:n=1  $ right foil
2222 2 -18.2731  -11        u=211 imp:n=1  $ left foil
2223 3 -7.4123  -13        u=211 imp:n=1  $ bottom clad
2224 3 -7.4123  -14        u=211 imp:n=1  $ top clad
2225 0             -15 10 11  u=211 imp:n=1  $ foil-plane void
2226 0             -22 12     u=211 imp:n=1  $ void around clad
2227 4 -2.7338  -20        u=211 imp:n=1  $ 1in graphite
2228 4 -2.7338  -21        u=211 imp:n=1  $ 2in graphite
2229 4 -2.7338  -30        u=211 imp:n=1  $ left rail
2230 4 -2.7338  -31        u=211 imp:n=1  $ right rail
2231 0            -23        u=211 imp:n=1  $ top gap
C  === fuel universe 212 slot (8, 1, 1) ===
2232 1 -19.7451  -10        u=212 imp:n=1  $ right foil
2233 2 -19.8212  -11        u=212 imp:n=1  $ left foil
2234 3 -7.3723  -13        u=212 imp:n=1  $ bottom clad
2235 3 -7.3723  -14        u=212 imp:n=1  $ top clad
2236 0             -15 10 11  u=212 imp:n=1  $ foil-plane void
2237 0             -22 12     u=212 imp:n=1  $ void around clad
2238 4 -2.7338  -20        u=212 imp:n=1  $ 1in graphite
2239 4 -2.7338  -21        u=212 imp:n=1  $ 2in graphite
2240 4 -2.7338  -30        u=212 imp:n=1  $ left rail
2241 4 -2.7338  -31        u=212 imp:n=1  $ right rail
2242 0            -23        u=212 imp:n=1  $ top gap
C  === fuel universe 213 slot (8, 3, 1) ===
2243 1 -19.1977  -10        u=213 imp:n=1  $ right foil
2244 2 -19.4106  -11        u=213 imp:n=1  $ left foil
2245 3 -7.0379  -13        u=213 imp:n=1  $ bottom clad
2246 3 -7.0379  -14        u=213 imp:n=1  $ top clad
2247 0             -15 10 11  u=213 imp:n=1  $ foil-plane void
2248 0             -22 12     u=213 imp:n=1  $ void around clad
2249 4 -2.7338  -20        u=213 imp:n=1  $ 1in graphite
2250 4 -2.7338  -21        u=213 imp:n=1  $ 2in graphite
2251 4 -2.7338  -30        u=213 imp:n=1  $ left rail
2252 4 -2.7338  -31        u=213 imp:n=1  $ right rail
2253 0            -23        u=213 imp:n=1  $ top gap
C  === fuel universe 214 slot (8, 5, 1) ===
2254 1 -19.3649  -10        u=214 imp:n=1  $ right foil
2255 2 -19.3649  -11        u=214 imp:n=1  $ left foil
2256 3 -7.3551  -13        u=214 imp:n=1  $ bottom clad
2257 3 -7.3551  -14        u=214 imp:n=1  $ top clad
2258 0             -15 10 11  u=214 imp:n=1  $ foil-plane void
2259 0             -22 12     u=214 imp:n=1  $ void around clad
2260 4 -2.7338  -20        u=214 imp:n=1  $ 1in graphite
2261 4 -2.7338  -21        u=214 imp:n=1  $ 2in graphite
2262 4 -2.7338  -30        u=214 imp:n=1  $ left rail
2263 4 -2.7338  -31        u=214 imp:n=1  $ right rail
2264 0            -23        u=214 imp:n=1  $ top gap
C  === fuel universe 215 slot (8, 7, 1) ===
2265 1 -18.9574  -10        u=215 imp:n=1  $ right foil
2266 2 -18.6837  -11        u=215 imp:n=1  $ left foil
2267 3 -7.1882  -13        u=215 imp:n=1  $ bottom clad
2268 3 -7.1882  -14        u=215 imp:n=1  $ top clad
2269 0             -15 10 11  u=215 imp:n=1  $ foil-plane void
2270 0             -22 12     u=215 imp:n=1  $ void around clad
2271 4 -2.7338  -20        u=215 imp:n=1  $ 1in graphite
2272 4 -2.7338  -21        u=215 imp:n=1  $ 2in graphite
2273 4 -2.7338  -30        u=215 imp:n=1  $ left rail
2274 4 -2.7338  -31        u=215 imp:n=1  $ right rail
2275 0            -23        u=215 imp:n=1  $ top gap
C  === fuel universe 216 slot (8, 2, 2) ===
2276 1 -18.5772  -10        u=216 imp:n=1  $ right foil
2277 2 -18.5468  -11        u=216 imp:n=1  $ left foil
2278 3 -7.2707  -13        u=216 imp:n=1  $ bottom clad
2279 3 -7.2707  -14        u=216 imp:n=1  $ top clad
2280 0             -15 10 11  u=216 imp:n=1  $ foil-plane void
2281 0             -22 12     u=216 imp:n=1  $ void around clad
2282 4 -2.7338  -20        u=216 imp:n=1  $ 1in graphite
2283 4 -2.7338  -21        u=216 imp:n=1  $ 2in graphite
2284 4 -2.7338  -30        u=216 imp:n=1  $ left rail
2285 4 -2.7338  -31        u=216 imp:n=1  $ right rail
2286 0            -23        u=216 imp:n=1  $ top gap
C  === fuel universe 217 slot (8, 4, 2) ===
2287 1 -18.9422  -10        u=217 imp:n=1  $ right foil
2288 2 -18.6228  -11        u=217 imp:n=1  $ left foil
2289 3 -7.3434  -13        u=217 imp:n=1  $ bottom clad
2290 3 -7.3434  -14        u=217 imp:n=1  $ top clad
2291 0             -15 10 11  u=217 imp:n=1  $ foil-plane void
2292 0             -22 12     u=217 imp:n=1  $ void around clad
2293 4 -2.7338  -20        u=217 imp:n=1  $ 1in graphite
2294 4 -2.7338  -21        u=217 imp:n=1  $ 2in graphite
2295 4 -2.7338  -30        u=217 imp:n=1  $ left rail
2296 4 -2.7338  -31        u=217 imp:n=1  $ right rail
2297 0            -23        u=217 imp:n=1  $ top gap
C  === fuel universe 218 slot (8, 6, 2) ===
2298 1 -18.8053  -10        u=218 imp:n=1  $ right foil
2299 2 -18.8205  -11        u=218 imp:n=1  $ left foil
2300 3 -7.2606  -13        u=218 imp:n=1  $ bottom clad
2301 3 -7.2606  -14        u=218 imp:n=1  $ top clad
2302 0             -15 10 11  u=218 imp:n=1  $ foil-plane void
2303 0             -22 12     u=218 imp:n=1  $ void around clad
2304 4 -2.7338  -20        u=218 imp:n=1  $ 1in graphite
2305 4 -2.7338  -21        u=218 imp:n=1  $ 2in graphite
2306 4 -2.7338  -30        u=218 imp:n=1  $ left rail
2307 4 -2.7338  -31        u=218 imp:n=1  $ right rail
2308 0            -23        u=218 imp:n=1  $ top gap
C  === fuel universe 219 slot (8, 8, 2) ===
2309 1 -18.4708  -10        u=219 imp:n=1  $ right foil
2310 2 -19.4714  -11        u=219 imp:n=1  $ left foil
2311 3 -7.3268  -13        u=219 imp:n=1  $ bottom clad
2312 3 -7.3268  -14        u=219 imp:n=1  $ top clad
2313 0             -15 10 11  u=219 imp:n=1  $ foil-plane void
2314 0             -22 12     u=219 imp:n=1  $ void around clad
2315 4 -2.7338  -20        u=219 imp:n=1  $ 1in graphite
2316 4 -2.7338  -21        u=219 imp:n=1  $ 2in graphite
2317 4 -2.7338  -30        u=219 imp:n=1  $ left rail
2318 4 -2.7338  -31        u=219 imp:n=1  $ right rail
2319 0            -23        u=219 imp:n=1  $ top gap
C  === fuel universe 220 slot (8, 1, 3) ===
2320 1 -19.958  -10        u=220 imp:n=1  $ right foil
2321 2 -19.7907  -11        u=220 imp:n=1  $ left foil
2322 3 -7.262  -13        u=220 imp:n=1  $ bottom clad
2323 3 -7.262  -14        u=220 imp:n=1  $ top clad
2324 0             -15 10 11  u=220 imp:n=1  $ foil-plane void
2325 0             -22 12     u=220 imp:n=1  $ void around clad
2326 4 -2.7338  -20        u=220 imp:n=1  $ 1in graphite
2327 4 -2.7338  -21        u=220 imp:n=1  $ 2in graphite
2328 4 -2.7338  -30        u=220 imp:n=1  $ left rail
2329 4 -2.7338  -31        u=220 imp:n=1  $ right rail
2330 0            -23        u=220 imp:n=1  $ top gap
C  === fuel universe 221 slot (8, 3, 3) ===
2331 1 -19.441  -10        u=221 imp:n=1  $ right foil
2332 2 -19.2281  -11        u=221 imp:n=1  $ left foil
2333 3 -7.0241  -13        u=221 imp:n=1  $ bottom clad
2334 3 -7.0241  -14        u=221 imp:n=1  $ top clad
2335 0             -15 10 11  u=221 imp:n=1  $ foil-plane void
2336 0             -22 12     u=221 imp:n=1  $ void around clad
2337 4 -2.7338  -20        u=221 imp:n=1  $ 1in graphite
2338 4 -2.7338  -21        u=221 imp:n=1  $ 2in graphite
2339 4 -2.7338  -30        u=221 imp:n=1  $ left rail
2340 4 -2.7338  -31        u=221 imp:n=1  $ right rail
2341 0            -23        u=221 imp:n=1  $ top gap
C  === fuel universe 222 slot (8, 5, 3) ===
2342 1 -19.1672  -10        u=222 imp:n=1  $ right foil
2343 2 -19.3802  -11        u=222 imp:n=1  $ left foil
2344 3 -7.2707  -13        u=222 imp:n=1  $ bottom clad
2345 3 -7.2707  -14        u=222 imp:n=1  $ top clad
2346 0             -15 10 11  u=222 imp:n=1  $ foil-plane void
2347 0             -22 12     u=222 imp:n=1  $ void around clad
2348 4 -2.7338  -20        u=222 imp:n=1  $ 1in graphite
2349 4 -2.7338  -21        u=222 imp:n=1  $ 2in graphite
2350 4 -2.7338  -30        u=222 imp:n=1  $ left rail
2351 4 -2.7338  -31        u=222 imp:n=1  $ right rail
2352 0            -23        u=222 imp:n=1  $ top gap
C  === fuel universe 223 slot (8, 7, 3) ===
2353 1 -19.7451  -10        u=223 imp:n=1  $ right foil
2354 2 -19.3802  -11        u=223 imp:n=1  $ left foil
2355 3 -7.2707  -13        u=223 imp:n=1  $ bottom clad
2356 3 -7.2707  -14        u=223 imp:n=1  $ top clad
2357 0             -15 10 11  u=223 imp:n=1  $ foil-plane void
2358 0             -22 12     u=223 imp:n=1  $ void around clad
2359 4 -2.7338  -20        u=223 imp:n=1  $ 1in graphite
2360 4 -2.7338  -21        u=223 imp:n=1  $ 2in graphite
2361 4 -2.7338  -30        u=223 imp:n=1  $ left rail
2362 4 -2.7338  -31        u=223 imp:n=1  $ right rail
2363 0            -23        u=223 imp:n=1  $ top gap
C  === fuel universe 224 slot (8, 2, 4) ===
2364 1 -18.5772  -10        u=224 imp:n=1  $ right foil
2365 2 -18.5468  -11        u=224 imp:n=1  $ left foil
2366 3 -7.2707  -13        u=224 imp:n=1  $ bottom clad
2367 3 -7.2707  -14        u=224 imp:n=1  $ top clad
2368 0             -15 10 11  u=224 imp:n=1  $ foil-plane void
2369 0             -22 12     u=224 imp:n=1  $ void around clad
2370 4 -2.7338  -20        u=224 imp:n=1  $ 1in graphite
2371 4 -2.7338  -21        u=224 imp:n=1  $ 2in graphite
2372 4 -2.7338  -30        u=224 imp:n=1  $ left rail
2373 4 -2.7338  -31        u=224 imp:n=1  $ right rail
2374 0            -23        u=224 imp:n=1  $ top gap
C  === fuel universe 225 slot (8, 4, 4) ===
2375 1 -18.9422  -10        u=225 imp:n=1  $ right foil
2376 2 -18.6228  -11        u=225 imp:n=1  $ left foil
2377 3 -7.3434  -13        u=225 imp:n=1  $ bottom clad
2378 3 -7.3434  -14        u=225 imp:n=1  $ top clad
2379 0             -15 10 11  u=225 imp:n=1  $ foil-plane void
2380 0             -22 12     u=225 imp:n=1  $ void around clad
2381 4 -2.7338  -20        u=225 imp:n=1  $ 1in graphite
2382 4 -2.7338  -21        u=225 imp:n=1  $ 2in graphite
2383 4 -2.7338  -30        u=225 imp:n=1  $ left rail
2384 4 -2.7338  -31        u=225 imp:n=1  $ right rail
2385 0            -23        u=225 imp:n=1  $ top gap
C  === fuel universe 226 slot (8, 6, 4) ===
2386 1 -18.8053  -10        u=226 imp:n=1  $ right foil
2387 2 -18.8205  -11        u=226 imp:n=1  $ left foil
2388 3 -7.2606  -13        u=226 imp:n=1  $ bottom clad
2389 3 -7.2606  -14        u=226 imp:n=1  $ top clad
2390 0             -15 10 11  u=226 imp:n=1  $ foil-plane void
2391 0             -22 12     u=226 imp:n=1  $ void around clad
2392 4 -2.7338  -20        u=226 imp:n=1  $ 1in graphite
2393 4 -2.7338  -21        u=226 imp:n=1  $ 2in graphite
2394 4 -2.7338  -30        u=226 imp:n=1  $ left rail
2395 4 -2.7338  -31        u=226 imp:n=1  $ right rail
2396 0            -23        u=226 imp:n=1  $ top gap
C  === fuel universe 227 slot (8, 8, 4) ===
2397 1 -18.4708  -10        u=227 imp:n=1  $ right foil
2398 2 -19.4714  -11        u=227 imp:n=1  $ left foil
2399 3 -7.3268  -13        u=227 imp:n=1  $ bottom clad
2400 3 -7.3268  -14        u=227 imp:n=1  $ top clad
2401 0             -15 10 11  u=227 imp:n=1  $ foil-plane void
2402 0             -22 12     u=227 imp:n=1  $ void around clad
2403 4 -2.7338  -20        u=227 imp:n=1  $ 1in graphite
2404 4 -2.7338  -21        u=227 imp:n=1  $ 2in graphite
2405 4 -2.7338  -30        u=227 imp:n=1  $ left rail
2406 4 -2.7338  -31        u=227 imp:n=1  $ right rail
2407 0            -23        u=227 imp:n=1  $ top gap
C  === fuel universe 228 slot (9, 2, 1) ===
2408 1 -17.1629  -10        u=228 imp:n=1  $ right foil
2409 2 -17.239  -11        u=228 imp:n=1  $ left foil
2410 3 -7.371  -13        u=228 imp:n=1  $ bottom clad
2411 3 -7.371  -14        u=228 imp:n=1  $ top clad
2412 0             -15 10 11  u=228 imp:n=1  $ foil-plane void
2413 0             -22 12     u=228 imp:n=1  $ void around clad
2414 4 -2.7338  -20        u=228 imp:n=1  $ 1in graphite
2415 4 -2.7338  -21        u=228 imp:n=1  $ 2in graphite
2416 4 -2.7338  -30        u=228 imp:n=1  $ left rail
2417 4 -2.7338  -31        u=228 imp:n=1  $ right rail
2418 0            -23        u=228 imp:n=1  $ top gap
C  === fuel universe 229 slot (9, 4, 1) ===
2419 1 -19.6083  -10        u=229 imp:n=1  $ right foil
2420 2 -19.3193  -11        u=229 imp:n=1  $ left foil
2421 3 -7.0344  -13        u=229 imp:n=1  $ bottom clad
2422 3 -7.0344  -14        u=229 imp:n=1  $ top clad
2423 0             -15 10 11  u=229 imp:n=1  $ foil-plane void
2424 0             -22 12     u=229 imp:n=1  $ void around clad
2425 4 -2.7338  -20        u=229 imp:n=1  $ 1in graphite
2426 4 -2.7338  -21        u=229 imp:n=1  $ 2in graphite
2427 4 -2.7338  -30        u=229 imp:n=1  $ left rail
2428 4 -2.7338  -31        u=229 imp:n=1  $ right rail
2429 0            -23        u=229 imp:n=1  $ top gap
C  === fuel universe 230 slot (9, 6, 1) ===
2430 1 -18.5924  -10        u=230 imp:n=1  $ right foil
2431 2 -18.7445  -11        u=230 imp:n=1  $ left foil
2432 3 -7.3951  -13        u=230 imp:n=1  $ bottom clad
2433 3 -7.3951  -14        u=230 imp:n=1  $ top clad
2434 0             -15 10 11  u=230 imp:n=1  $ foil-plane void
2435 0             -22 12     u=230 imp:n=1  $ void around clad
2436 4 -2.7338  -20        u=230 imp:n=1  $ 1in graphite
2437 4 -2.7338  -21        u=230 imp:n=1  $ 2in graphite
2438 4 -2.7338  -30        u=230 imp:n=1  $ left rail
2439 4 -2.7338  -31        u=230 imp:n=1  $ right rail
2440 0            -23        u=230 imp:n=1  $ top gap
C  === fuel universe 231 slot (9, 8, 1) ===
2441 1 -17.2542  -10        u=231 imp:n=1  $ right foil
2442 2 -17.2542  -11        u=231 imp:n=1  $ left foil
2443 3 -7.4158  -13        u=231 imp:n=1  $ bottom clad
2444 3 -7.4158  -14        u=231 imp:n=1  $ top clad
2445 0             -15 10 11  u=231 imp:n=1  $ foil-plane void
2446 0             -22 12     u=231 imp:n=1  $ void around clad
2447 4 -2.7338  -20        u=231 imp:n=1  $ 1in graphite
2448 4 -2.7338  -21        u=231 imp:n=1  $ 2in graphite
2449 4 -2.7338  -30        u=231 imp:n=1  $ left rail
2450 4 -2.7338  -31        u=231 imp:n=1  $ right rail
2451 0            -23        u=231 imp:n=1  $ top gap
C  === fuel universe 232 slot (9, 1, 2) ===
2452 1 -19.2889  -10        u=232 imp:n=1  $ right foil
2453 2 -18.7901  -11        u=232 imp:n=1  $ left foil
2454 3 -7.2268  -13        u=232 imp:n=1  $ bottom clad
2455 3 -7.2268  -14        u=232 imp:n=1  $ top clad
2456 0             -15 10 11  u=232 imp:n=1  $ foil-plane void
2457 0             -22 12     u=232 imp:n=1  $ void around clad
2458 4 -2.7338  -20        u=232 imp:n=1  $ 1in graphite
2459 4 -2.7338  -21        u=232 imp:n=1  $ 2in graphite
2460 4 -2.7338  -30        u=232 imp:n=1  $ left rail
2461 4 -2.7338  -31        u=232 imp:n=1  $ right rail
2462 0            -23        u=232 imp:n=1  $ top gap
C  === fuel universe 233 slot (9, 3, 2) ===
2463 1 -19.2433  -10        u=233 imp:n=1  $ right foil
2464 2 -18.5012  -11        u=233 imp:n=1  $ left foil
2465 3 -7.4406  -13        u=233 imp:n=1  $ bottom clad
2466 3 -7.4406  -14        u=233 imp:n=1  $ top clad
2467 0             -15 10 11  u=233 imp:n=1  $ foil-plane void
2468 0             -22 12     u=233 imp:n=1  $ void around clad
2469 4 -2.7338  -20        u=233 imp:n=1  $ 1in graphite
2470 4 -2.7338  -21        u=233 imp:n=1  $ 2in graphite
2471 4 -2.7338  -30        u=233 imp:n=1  $ left rail
2472 4 -2.7338  -31        u=233 imp:n=1  $ right rail
2473 0            -23        u=233 imp:n=1  $ top gap
C  === fuel universe 234 slot (9, 5, 2) ===
2474 1 -18.6532  -10        u=234 imp:n=1  $ right foil
2475 2 -18.927  -11        u=234 imp:n=1  $ left foil
2476 3 -7.202  -13        u=234 imp:n=1  $ bottom clad
2477 3 -7.202  -14        u=234 imp:n=1  $ top clad
2478 0             -15 10 11  u=234 imp:n=1  $ foil-plane void
2479 0             -22 12     u=234 imp:n=1  $ void around clad
2480 4 -2.7338  -20        u=234 imp:n=1  $ 1in graphite
2481 4 -2.7338  -21        u=234 imp:n=1  $ 2in graphite
2482 4 -2.7338  -30        u=234 imp:n=1  $ left rail
2483 4 -2.7338  -31        u=234 imp:n=1  $ right rail
2484 0            -23        u=234 imp:n=1  $ top gap
C  === fuel universe 235 slot (9, 7, 2) ===
2485 1 -18.8509  -10        u=235 imp:n=1  $ right foil
2486 2 -19.076  -11        u=235 imp:n=1  $ left foil
2487 3 -7.2707  -13        u=235 imp:n=1  $ bottom clad
2488 3 -7.2707  -14        u=235 imp:n=1  $ top clad
2489 0             -15 10 11  u=235 imp:n=1  $ foil-plane void
2490 0             -22 12     u=235 imp:n=1  $ void around clad
2491 4 -2.7338  -20        u=235 imp:n=1  $ 1in graphite
2492 4 -2.7338  -21        u=235 imp:n=1  $ 2in graphite
2493 4 -2.7338  -30        u=235 imp:n=1  $ left rail
2494 4 -2.7338  -31        u=235 imp:n=1  $ right rail
2495 0            -23        u=235 imp:n=1  $ top gap
C  === fuel universe 236 slot (9, 2, 3) ===
2496 1 -18.1514  -10        u=236 imp:n=1  $ right foil
2497 2 -17.0565  -11        u=236 imp:n=1  $ left foil
2498 3 -7.2572  -13        u=236 imp:n=1  $ bottom clad
2499 3 -7.2572  -14        u=236 imp:n=1  $ top clad
2500 0             -15 10 11  u=236 imp:n=1  $ foil-plane void
2501 0             -22 12     u=236 imp:n=1  $ void around clad
2502 4 -2.7338  -20        u=236 imp:n=1  $ 1in graphite
2503 4 -2.7338  -21        u=236 imp:n=1  $ 2in graphite
2504 4 -2.7338  -30        u=236 imp:n=1  $ left rail
2505 4 -2.7338  -31        u=236 imp:n=1  $ right rail
2506 0            -23        u=236 imp:n=1  $ top gap
C  === fuel universe 237 slot (9, 4, 3) ===
2507 1 -19.6691  -10        u=237 imp:n=1  $ right foil
2508 2 -19.517  -11        u=237 imp:n=1  $ left foil
2509 3 -7.0448  -13        u=237 imp:n=1  $ bottom clad
2510 3 -7.0448  -14        u=237 imp:n=1  $ top clad
2511 0             -15 10 11  u=237 imp:n=1  $ foil-plane void
2512 0             -22 12     u=237 imp:n=1  $ void around clad
2513 4 -2.7338  -20        u=237 imp:n=1  $ 1in graphite
2514 4 -2.7338  -21        u=237 imp:n=1  $ 2in graphite
2515 4 -2.7338  -30        u=237 imp:n=1  $ left rail
2516 4 -2.7338  -31        u=237 imp:n=1  $ right rail
2517 0            -23        u=237 imp:n=1  $ top gap
C  === fuel universe 238 slot (9, 6, 3) ===
2518 1 -19.8668  -10        u=238 imp:n=1  $ right foil
2519 2 -19.8212  -11        u=238 imp:n=1  $ left foil
2520 3 -7.0689  -13        u=238 imp:n=1  $ bottom clad
2521 3 -7.0689  -14        u=238 imp:n=1  $ top clad
2522 0             -15 10 11  u=238 imp:n=1  $ foil-plane void
2523 0             -22 12     u=238 imp:n=1  $ void around clad
2524 4 -2.7338  -20        u=238 imp:n=1  $ 1in graphite
2525 4 -2.7338  -21        u=238 imp:n=1  $ 2in graphite
2526 4 -2.7338  -30        u=238 imp:n=1  $ left rail
2527 4 -2.7338  -31        u=238 imp:n=1  $ right rail
2528 0            -23        u=238 imp:n=1  $ top gap
C  === fuel universe 239 slot (9, 8, 3) ===
2529 1 -17.3454  -10        u=239 imp:n=1  $ right foil
2530 2 -17.5583  -11        u=239 imp:n=1  $ left foil
2531 3 -7.2572  -13        u=239 imp:n=1  $ bottom clad
2532 3 -7.2572  -14        u=239 imp:n=1  $ top clad
2533 0             -15 10 11  u=239 imp:n=1  $ foil-plane void
2534 0             -22 12     u=239 imp:n=1  $ void around clad
2535 4 -2.7338  -20        u=239 imp:n=1  $ 1in graphite
2536 4 -2.7338  -21        u=239 imp:n=1  $ 2in graphite
2537 4 -2.7338  -30        u=239 imp:n=1  $ left rail
2538 4 -2.7338  -31        u=239 imp:n=1  $ right rail
2539 0            -23        u=239 imp:n=1  $ top gap
C  === fuel universe 240 slot (9, 1, 4) ===
2540 1 -19.2889  -10        u=240 imp:n=1  $ right foil
2541 2 -18.7901  -11        u=240 imp:n=1  $ left foil
2542 3 -7.2268  -13        u=240 imp:n=1  $ bottom clad
2543 3 -7.2268  -14        u=240 imp:n=1  $ top clad
2544 0             -15 10 11  u=240 imp:n=1  $ foil-plane void
2545 0             -22 12     u=240 imp:n=1  $ void around clad
2546 4 -2.7338  -20        u=240 imp:n=1  $ 1in graphite
2547 4 -2.7338  -21        u=240 imp:n=1  $ 2in graphite
2548 4 -2.7338  -30        u=240 imp:n=1  $ left rail
2549 4 -2.7338  -31        u=240 imp:n=1  $ right rail
2550 0            -23        u=240 imp:n=1  $ top gap
C  === fuel universe 241 slot (9, 3, 4) ===
2551 1 -19.2433  -10        u=241 imp:n=1  $ right foil
2552 2 -18.5012  -11        u=241 imp:n=1  $ left foil
2553 3 -7.4406  -13        u=241 imp:n=1  $ bottom clad
2554 3 -7.4406  -14        u=241 imp:n=1  $ top clad
2555 0             -15 10 11  u=241 imp:n=1  $ foil-plane void
2556 0             -22 12     u=241 imp:n=1  $ void around clad
2557 4 -2.7338  -20        u=241 imp:n=1  $ 1in graphite
2558 4 -2.7338  -21        u=241 imp:n=1  $ 2in graphite
2559 4 -2.7338  -30        u=241 imp:n=1  $ left rail
2560 4 -2.7338  -31        u=241 imp:n=1  $ right rail
2561 0            -23        u=241 imp:n=1  $ top gap
C  === fuel universe 242 slot (9, 5, 4) ===
2562 1 -18.6532  -10        u=242 imp:n=1  $ right foil
2563 2 -18.927  -11        u=242 imp:n=1  $ left foil
2564 3 -7.202  -13        u=242 imp:n=1  $ bottom clad
2565 3 -7.202  -14        u=242 imp:n=1  $ top clad
2566 0             -15 10 11  u=242 imp:n=1  $ foil-plane void
2567 0             -22 12     u=242 imp:n=1  $ void around clad
2568 4 -2.7338  -20        u=242 imp:n=1  $ 1in graphite
2569 4 -2.7338  -21        u=242 imp:n=1  $ 2in graphite
2570 4 -2.7338  -30        u=242 imp:n=1  $ left rail
2571 4 -2.7338  -31        u=242 imp:n=1  $ right rail
2572 0            -23        u=242 imp:n=1  $ top gap
C  === fuel universe 243 slot (9, 7, 4) ===
2573 1 -18.8509  -10        u=243 imp:n=1  $ right foil
2574 2 -19.076  -11        u=243 imp:n=1  $ left foil
2575 3 -7.2707  -13        u=243 imp:n=1  $ bottom clad
2576 3 -7.2707  -14        u=243 imp:n=1  $ top clad
2577 0             -15 10 11  u=243 imp:n=1  $ foil-plane void
2578 0             -22 12     u=243 imp:n=1  $ void around clad
2579 4 -2.7338  -20        u=243 imp:n=1  $ 1in graphite
2580 4 -2.7338  -21        u=243 imp:n=1  $ 2in graphite
2581 4 -2.7338  -30        u=243 imp:n=1  $ left rail
2582 4 -2.7338  -31        u=243 imp:n=1  $ right rail
2583 0            -23        u=243 imp:n=1  $ top gap
C  === fuel universe 244 slot (10, 1, 1) ===
2584 1 -17.9537  -10        u=244 imp:n=1  $ right foil
2585 2 -17.7712  -11        u=244 imp:n=1  $ left foil
2586 3 -7.3468  -13        u=244 imp:n=1  $ bottom clad
2587 3 -7.3468  -14        u=244 imp:n=1  $ top clad
2588 0             -15 10 11  u=244 imp:n=1  $ foil-plane void
2589 0             -22 12     u=244 imp:n=1  $ void around clad
2590 4 -2.7338  -20        u=244 imp:n=1  $ 1in graphite
2591 4 -2.7338  -21        u=244 imp:n=1  $ 2in graphite
2592 4 -2.7338  -30        u=244 imp:n=1  $ left rail
2593 4 -2.7338  -31        u=244 imp:n=1  $ right rail
2594 0            -23        u=244 imp:n=1  $ top gap
C  === fuel universe 245 slot (10, 3, 1) ===
2595 1 -17.9993  -10        u=245 imp:n=1  $ right foil
2596 2 -18.0145  -11        u=245 imp:n=1  $ left foil
2597 3 -7.4192  -13        u=245 imp:n=1  $ bottom clad
2598 3 -7.4192  -14        u=245 imp:n=1  $ top clad
2599 0             -15 10 11  u=245 imp:n=1  $ foil-plane void
2600 0             -22 12     u=245 imp:n=1  $ void around clad
2601 4 -2.7338  -20        u=245 imp:n=1  $ 1in graphite
2602 4 -2.7338  -21        u=245 imp:n=1  $ 2in graphite
2603 4 -2.7338  -30        u=245 imp:n=1  $ left rail
2604 4 -2.7338  -31        u=245 imp:n=1  $ right rail
2605 0            -23        u=245 imp:n=1  $ top gap
C  === fuel universe 246 slot (10, 5, 1) ===
2606 1 -18.0297  -10        u=246 imp:n=1  $ right foil
2607 2 -18.0602  -11        u=246 imp:n=1  $ left foil
2608 3 -7.2641  -13        u=246 imp:n=1  $ bottom clad
2609 3 -7.2641  -14        u=246 imp:n=1  $ top clad
2610 0             -15 10 11  u=246 imp:n=1  $ foil-plane void
2611 0             -22 12     u=246 imp:n=1  $ void around clad
2612 4 -2.7338  -20        u=246 imp:n=1  $ 1in graphite
2613 4 -2.7338  -21        u=246 imp:n=1  $ 2in graphite
2614 4 -2.7338  -30        u=246 imp:n=1  $ left rail
2615 4 -2.7338  -31        u=246 imp:n=1  $ right rail
2616 0            -23        u=246 imp:n=1  $ top gap
C  === fuel universe 247 slot (10, 7, 1) ===
2617 1 -17.8016  -10        u=247 imp:n=1  $ right foil
2618 2 -17.8777  -11        u=247 imp:n=1  $ left foil
2619 3 -7.1503  -13        u=247 imp:n=1  $ bottom clad
2620 3 -7.1503  -14        u=247 imp:n=1  $ top clad
2621 0             -15 10 11  u=247 imp:n=1  $ foil-plane void
2622 0             -22 12     u=247 imp:n=1  $ void around clad
2623 4 -2.7338  -20        u=247 imp:n=1  $ 1in graphite
2624 4 -2.7338  -21        u=247 imp:n=1  $ 2in graphite
2625 4 -2.7338  -30        u=247 imp:n=1  $ left rail
2626 4 -2.7338  -31        u=247 imp:n=1  $ right rail
2627 0            -23        u=247 imp:n=1  $ top gap
C  === fuel universe 248 slot (10, 2, 2) ===
2628 1 -18.4251  -10        u=248 imp:n=1  $ right foil
2629 2 -18.3339  -11        u=248 imp:n=1  $ left foil
2630 3 -7.2503  -13        u=248 imp:n=1  $ bottom clad
2631 3 -7.2503  -14        u=248 imp:n=1  $ top clad
2632 0             -15 10 11  u=248 imp:n=1  $ foil-plane void
2633 0             -22 12     u=248 imp:n=1  $ void around clad
2634 4 -2.7338  -20        u=248 imp:n=1  $ 1in graphite
2635 4 -2.7338  -21        u=248 imp:n=1  $ 2in graphite
2636 4 -2.7338  -30        u=248 imp:n=1  $ left rail
2637 4 -2.7338  -31        u=248 imp:n=1  $ right rail
2638 0            -23        u=248 imp:n=1  $ top gap
C  === fuel universe 249 slot (10, 4, 2) ===
2639 1 -18.4555  -10        u=249 imp:n=1  $ right foil
2640 2 -18.562  -11        u=249 imp:n=1  $ left foil
2641 3 -7.4675  -13        u=249 imp:n=1  $ bottom clad
2642 3 -7.4675  -14        u=249 imp:n=1  $ top clad
2643 0             -15 10 11  u=249 imp:n=1  $ foil-plane void
2644 0             -22 12     u=249 imp:n=1  $ void around clad
2645 4 -2.7338  -20        u=249 imp:n=1  $ 1in graphite
2646 4 -2.7338  -21        u=249 imp:n=1  $ 2in graphite
2647 4 -2.7338  -30        u=249 imp:n=1  $ left rail
2648 4 -2.7338  -31        u=249 imp:n=1  $ right rail
2649 0            -23        u=249 imp:n=1  $ top gap
C  === fuel universe 250 slot (10, 6, 2) ===
2650 1 -18.562  -10        u=250 imp:n=1  $ right foil
2651 2 -18.5012  -11        u=250 imp:n=1  $ left foil
2652 3 -7.1813  -13        u=250 imp:n=1  $ bottom clad
2653 3 -7.1813  -14        u=250 imp:n=1  $ top clad
2654 0             -15 10 11  u=250 imp:n=1  $ foil-plane void
2655 0             -22 12     u=250 imp:n=1  $ void around clad
2656 4 -2.7338  -20        u=250 imp:n=1  $ 1in graphite
2657 4 -2.7338  -21        u=250 imp:n=1  $ 2in graphite
2658 4 -2.7338  -30        u=250 imp:n=1  $ left rail
2659 4 -2.7338  -31        u=250 imp:n=1  $ right rail
2660 0            -23        u=250 imp:n=1  $ top gap
C  === fuel universe 251 slot (10, 8, 2) ===
2661 1 -19.0  -10        u=251 imp:n=1  $ right foil
2662 2 -17.68  -11        u=251 imp:n=1  $ left foil
2663 3 -7.1993  -13        u=251 imp:n=1  $ bottom clad
2664 3 -7.1993  -14        u=251 imp:n=1  $ top clad
2665 0             -15 10 11  u=251 imp:n=1  $ foil-plane void
2666 0             -22 12     u=251 imp:n=1  $ void around clad
2667 4 -2.7338  -20        u=251 imp:n=1  $ 1in graphite
2668 4 -2.7338  -21        u=251 imp:n=1  $ 2in graphite
2669 4 -2.7338  -30        u=251 imp:n=1  $ left rail
2670 4 -2.7338  -31        u=251 imp:n=1  $ right rail
2671 0            -23        u=251 imp:n=1  $ top gap
C  === fuel universe 252 slot (10, 1, 3) ===
2672 1 -17.6191  -10        u=252 imp:n=1  $ right foil
2673 2 -17.6191  -11        u=252 imp:n=1  $ left foil
2674 3 -7.2707  -13        u=252 imp:n=1  $ bottom clad
2675 3 -7.2707  -14        u=252 imp:n=1  $ top clad
2676 0             -15 10 11  u=252 imp:n=1  $ foil-plane void
2677 0             -22 12     u=252 imp:n=1  $ void around clad
2678 4 -2.7338  -20        u=252 imp:n=1  $ 1in graphite
2679 4 -2.7338  -21        u=252 imp:n=1  $ 2in graphite
2680 4 -2.7338  -30        u=252 imp:n=1  $ left rail
2681 4 -2.7338  -31        u=252 imp:n=1  $ right rail
2682 0            -23        u=252 imp:n=1  $ top gap
C  === fuel universe 253 slot (10, 3, 3) ===
2683 1 -17.391  -10        u=253 imp:n=1  $ right foil
2684 2 -18.5012  -11        u=253 imp:n=1  $ left foil
2685 3 -7.171  -13        u=253 imp:n=1  $ bottom clad
2686 3 -7.171  -14        u=253 imp:n=1  $ top clad
2687 0             -15 10 11  u=253 imp:n=1  $ foil-plane void
2688 0             -22 12     u=253 imp:n=1  $ void around clad
2689 4 -2.7338  -20        u=253 imp:n=1  $ 1in graphite
2690 4 -2.7338  -21        u=253 imp:n=1  $ 2in graphite
2691 4 -2.7338  -30        u=253 imp:n=1  $ left rail
2692 4 -2.7338  -31        u=253 imp:n=1  $ right rail
2693 0            -23        u=253 imp:n=1  $ top gap
C  === fuel universe 254 slot (10, 5, 3) ===
2694 1 -18.045  -10        u=254 imp:n=1  $ right foil
2695 2 -18.045  -11        u=254 imp:n=1  $ left foil
2696 3 -7.2707  -13        u=254 imp:n=1  $ bottom clad
2697 3 -7.2707  -14        u=254 imp:n=1  $ top clad
2698 0             -15 10 11  u=254 imp:n=1  $ foil-plane void
2699 0             -22 12     u=254 imp:n=1  $ void around clad
2700 4 -2.7338  -20        u=254 imp:n=1  $ 1in graphite
2701 4 -2.7338  -21        u=254 imp:n=1  $ 2in graphite
2702 4 -2.7338  -30        u=254 imp:n=1  $ left rail
2703 4 -2.7338  -31        u=254 imp:n=1  $ right rail
2704 0            -23        u=254 imp:n=1  $ top gap
C  === fuel universe 255 slot (10, 7, 3) ===
2705 1 -17.5735  -10        u=255 imp:n=1  $ right foil
2706 2 -17.6952  -11        u=255 imp:n=1  $ left foil
2707 3 -7.2707  -13        u=255 imp:n=1  $ bottom clad
2708 3 -7.2707  -14        u=255 imp:n=1  $ top clad
2709 0             -15 10 11  u=255 imp:n=1  $ foil-plane void
2710 0             -22 12     u=255 imp:n=1  $ void around clad
2711 4 -2.7338  -20        u=255 imp:n=1  $ 1in graphite
2712 4 -2.7338  -21        u=255 imp:n=1  $ 2in graphite
2713 4 -2.7338  -30        u=255 imp:n=1  $ left rail
2714 4 -2.7338  -31        u=255 imp:n=1  $ right rail
2715 0            -23        u=255 imp:n=1  $ top gap
C  === fuel universe 256 slot (10, 2, 4) ===
2716 1 -18.4251  -10        u=256 imp:n=1  $ right foil
2717 2 -18.3339  -11        u=256 imp:n=1  $ left foil
2718 3 -7.2503  -13        u=256 imp:n=1  $ bottom clad
2719 3 -7.2503  -14        u=256 imp:n=1  $ top clad
2720 0             -15 10 11  u=256 imp:n=1  $ foil-plane void
2721 0             -22 12     u=256 imp:n=1  $ void around clad
2722 4 -2.7338  -20        u=256 imp:n=1  $ 1in graphite
2723 4 -2.7338  -21        u=256 imp:n=1  $ 2in graphite
2724 4 -2.7338  -30        u=256 imp:n=1  $ left rail
2725 4 -2.7338  -31        u=256 imp:n=1  $ right rail
2726 0            -23        u=256 imp:n=1  $ top gap
C  === fuel universe 257 slot (10, 4, 4) ===
2727 1 -18.4555  -10        u=257 imp:n=1  $ right foil
2728 2 -18.562  -11        u=257 imp:n=1  $ left foil
2729 3 -7.4675  -13        u=257 imp:n=1  $ bottom clad
2730 3 -7.4675  -14        u=257 imp:n=1  $ top clad
2731 0             -15 10 11  u=257 imp:n=1  $ foil-plane void
2732 0             -22 12     u=257 imp:n=1  $ void around clad
2733 4 -2.7338  -20        u=257 imp:n=1  $ 1in graphite
2734 4 -2.7338  -21        u=257 imp:n=1  $ 2in graphite
2735 4 -2.7338  -30        u=257 imp:n=1  $ left rail
2736 4 -2.7338  -31        u=257 imp:n=1  $ right rail
2737 0            -23        u=257 imp:n=1  $ top gap
C  === fuel universe 258 slot (10, 6, 4) ===
2738 1 -18.562  -10        u=258 imp:n=1  $ right foil
2739 2 -18.5012  -11        u=258 imp:n=1  $ left foil
2740 3 -7.1813  -13        u=258 imp:n=1  $ bottom clad
2741 3 -7.1813  -14        u=258 imp:n=1  $ top clad
2742 0             -15 10 11  u=258 imp:n=1  $ foil-plane void
2743 0             -22 12     u=258 imp:n=1  $ void around clad
2744 4 -2.7338  -20        u=258 imp:n=1  $ 1in graphite
2745 4 -2.7338  -21        u=258 imp:n=1  $ 2in graphite
2746 4 -2.7338  -30        u=258 imp:n=1  $ left rail
2747 4 -2.7338  -31        u=258 imp:n=1  $ right rail
2748 0            -23        u=258 imp:n=1  $ top gap
C  === fuel universe 259 slot (10, 8, 4) ===
2749 1 -19.0  -10        u=259 imp:n=1  $ right foil
2750 2 -17.68  -11        u=259 imp:n=1  $ left foil
2751 3 -7.1993  -13        u=259 imp:n=1  $ bottom clad
2752 3 -7.1993  -14        u=259 imp:n=1  $ top clad
2753 0             -15 10 11  u=259 imp:n=1  $ foil-plane void
2754 0             -22 12     u=259 imp:n=1  $ void around clad
2755 4 -2.7338  -20        u=259 imp:n=1  $ 1in graphite
2756 4 -2.7338  -21        u=259 imp:n=1  $ 2in graphite
2757 4 -2.7338  -30        u=259 imp:n=1  $ left rail
2758 4 -2.7338  -31        u=259 imp:n=1  $ right rail
2759 0            -23        u=259 imp:n=1  $ top gap
C  === fuel universe 260 slot (11, 2, 1) ===
2760 1 -17.2238  -10        u=260 imp:n=1  $ right foil
2761 2 -17.2542  -11        u=260 imp:n=1  $ left foil
2762 3 -7.4227  -13        u=260 imp:n=1  $ bottom clad
2763 3 -7.4227  -14        u=260 imp:n=1  $ top clad
2764 0             -15 10 11  u=260 imp:n=1  $ foil-plane void
2765 0             -22 12     u=260 imp:n=1  $ void around clad
2766 4 -2.7338  -20        u=260 imp:n=1  $ 1in graphite
2767 4 -2.7338  -21        u=260 imp:n=1  $ 2in graphite
2768 4 -2.7338  -30        u=260 imp:n=1  $ left rail
2769 4 -2.7338  -31        u=260 imp:n=1  $ right rail
2770 0            -23        u=260 imp:n=1  $ top gap
C  === fuel universe 261 slot (11, 4, 1) ===
2771 1 -19.1977  -10        u=261 imp:n=1  $ right foil
2772 2 -19.1977  -11        u=261 imp:n=1  $ left foil
2773 3 -7.431  -13        u=261 imp:n=1  $ bottom clad
2774 3 -7.431  -14        u=261 imp:n=1  $ top clad
2775 0             -15 10 11  u=261 imp:n=1  $ foil-plane void
2776 0             -22 12     u=261 imp:n=1  $ void around clad
2777 4 -2.7338  -20        u=261 imp:n=1  $ 1in graphite
2778 4 -2.7338  -21        u=261 imp:n=1  $ 2in graphite
2779 4 -2.7338  -30        u=261 imp:n=1  $ left rail
2780 4 -2.7338  -31        u=261 imp:n=1  $ right rail
2781 0            -23        u=261 imp:n=1  $ top gap
C  === fuel universe 262 slot (11, 6, 1) ===
2782 1 -19.2889  -10        u=262 imp:n=1  $ right foil
2783 2 -18.9422  -11        u=262 imp:n=1  $ left foil
2784 3 -7.2707  -13        u=262 imp:n=1  $ bottom clad
2785 3 -7.2707  -14        u=262 imp:n=1  $ top clad
2786 0             -15 10 11  u=262 imp:n=1  $ foil-plane void
2787 0             -22 12     u=262 imp:n=1  $ void around clad
2788 4 -2.7338  -20        u=262 imp:n=1  $ 1in graphite
2789 4 -2.7338  -21        u=262 imp:n=1  $ 2in graphite
2790 4 -2.7338  -30        u=262 imp:n=1  $ left rail
2791 4 -2.7338  -31        u=262 imp:n=1  $ right rail
2792 0            -23        u=262 imp:n=1  $ top gap
C  === fuel universe 263 slot (11, 8, 1) ===
2793 1 -19.152  -10        u=263 imp:n=1  $ right foil
2794 2 -19.2129  -11        u=263 imp:n=1  $ left foil
2795 3 -7.231  -13        u=263 imp:n=1  $ bottom clad
2796 3 -7.231  -14        u=263 imp:n=1  $ top clad
2797 0             -15 10 11  u=263 imp:n=1  $ foil-plane void
2798 0             -22 12     u=263 imp:n=1  $ void around clad
2799 4 -2.7338  -20        u=263 imp:n=1  $ 1in graphite
2800 4 -2.7338  -21        u=263 imp:n=1  $ 2in graphite
2801 4 -2.7338  -30        u=263 imp:n=1  $ left rail
2802 4 -2.7338  -31        u=263 imp:n=1  $ right rail
2803 0            -23        u=263 imp:n=1  $ top gap
C  === fuel universe 264 slot (11, 1, 2) ===
2804 1 -18.2883  -10        u=264 imp:n=1  $ right foil
2805 2 -18.2122  -11        u=264 imp:n=1  $ left foil
2806 3 -7.2399  -13        u=264 imp:n=1  $ bottom clad
2807 3 -7.2399  -14        u=264 imp:n=1  $ top clad
2808 0             -15 10 11  u=264 imp:n=1  $ foil-plane void
2809 0             -22 12     u=264 imp:n=1  $ void around clad
2810 4 -2.7338  -20        u=264 imp:n=1  $ 1in graphite
2811 4 -2.7338  -21        u=264 imp:n=1  $ 2in graphite
2812 4 -2.7338  -30        u=264 imp:n=1  $ left rail
2813 4 -2.7338  -31        u=264 imp:n=1  $ right rail
2814 0            -23        u=264 imp:n=1  $ top gap
C  === fuel universe 265 slot (11, 3, 2) ===
2815 1 -18.3339  -10        u=265 imp:n=1  $ right foil
2816 2 -18.4555  -11        u=265 imp:n=1  $ left foil
2817 3 -7.2434  -13        u=265 imp:n=1  $ bottom clad
2818 3 -7.2434  -14        u=265 imp:n=1  $ top clad
2819 0             -15 10 11  u=265 imp:n=1  $ foil-plane void
2820 0             -22 12     u=265 imp:n=1  $ void around clad
2821 4 -2.7338  -20        u=265 imp:n=1  $ 1in graphite
2822 4 -2.7338  -21        u=265 imp:n=1  $ 2in graphite
2823 4 -2.7338  -30        u=265 imp:n=1  $ left rail
2824 4 -2.7338  -31        u=265 imp:n=1  $ right rail
2825 0            -23        u=265 imp:n=1  $ top gap
C  === fuel universe 266 slot (11, 5, 2) ===
2826 1 -18.3947  -10        u=266 imp:n=1  $ right foil
2827 2 -18.4099  -11        u=266 imp:n=1  $ left foil
2828 3 -7.171  -13        u=266 imp:n=1  $ bottom clad
2829 3 -7.171  -14        u=266 imp:n=1  $ top clad
2830 0             -15 10 11  u=266 imp:n=1  $ foil-plane void
2831 0             -22 12     u=266 imp:n=1  $ void around clad
2832 4 -2.7338  -20        u=266 imp:n=1  $ 1in graphite
2833 4 -2.7338  -21        u=266 imp:n=1  $ 2in graphite
2834 4 -2.7338  -30        u=266 imp:n=1  $ left rail
2835 4 -2.7338  -31        u=266 imp:n=1  $ right rail
2836 0            -23        u=266 imp:n=1  $ top gap
C  === fuel universe 267 slot (11, 7, 2) ===
2837 1 -18.2883  -10        u=267 imp:n=1  $ right foil
2838 2 -18.5012  -11        u=267 imp:n=1  $ left foil
2839 3 -7.4503  -13        u=267 imp:n=1  $ bottom clad
2840 3 -7.4503  -14        u=267 imp:n=1  $ top clad
2841 0             -15 10 11  u=267 imp:n=1  $ foil-plane void
2842 0             -22 12     u=267 imp:n=1  $ void around clad
2843 4 -2.7338  -20        u=267 imp:n=1  $ 1in graphite
2844 4 -2.7338  -21        u=267 imp:n=1  $ 2in graphite
2845 4 -2.7338  -30        u=267 imp:n=1  $ left rail
2846 4 -2.7338  -31        u=267 imp:n=1  $ right rail
2847 0            -23        u=267 imp:n=1  $ top gap
C  === fuel universe 268 slot (11, 2, 3) ===
2848 1 -19.1064  -10        u=268 imp:n=1  $ right foil
2849 2 -19.3497  -11        u=268 imp:n=1  $ left foil
2850 3 -7.0724  -13        u=268 imp:n=1  $ bottom clad
2851 3 -7.0724  -14        u=268 imp:n=1  $ top clad
2852 0             -15 10 11  u=268 imp:n=1  $ foil-plane void
2853 0             -22 12     u=268 imp:n=1  $ void around clad
2854 4 -2.7338  -20        u=268 imp:n=1  $ 1in graphite
2855 4 -2.7338  -21        u=268 imp:n=1  $ 2in graphite
2856 4 -2.7338  -30        u=268 imp:n=1  $ left rail
2857 4 -2.7338  -31        u=268 imp:n=1  $ right rail
2858 0            -23        u=268 imp:n=1  $ top gap
C  === fuel universe 269 slot (11, 4, 3) ===
2859 1 -19.0912  -10        u=269 imp:n=1  $ right foil
2860 2 -19.2737  -11        u=269 imp:n=1  $ left foil
2861 3 -7.162  -13        u=269 imp:n=1  $ bottom clad
2862 3 -7.162  -14        u=269 imp:n=1  $ top clad
2863 0             -15 10 11  u=269 imp:n=1  $ foil-plane void
2864 0             -22 12     u=269 imp:n=1  $ void around clad
2865 4 -2.7338  -20        u=269 imp:n=1  $ 1in graphite
2866 4 -2.7338  -21        u=269 imp:n=1  $ 2in graphite
2867 4 -2.7338  -30        u=269 imp:n=1  $ left rail
2868 4 -2.7338  -31        u=269 imp:n=1  $ right rail
2869 0            -23        u=269 imp:n=1  $ top gap
C  === fuel universe 270 slot (11, 6, 3) ===
2870 1 -19.003  -10        u=270 imp:n=1  $ right foil
2871 2 -18.9574  -11        u=270 imp:n=1  $ left foil
2872 3 -7.2537  -13        u=270 imp:n=1  $ bottom clad
2873 3 -7.2537  -14        u=270 imp:n=1  $ top clad
2874 0             -15 10 11  u=270 imp:n=1  $ foil-plane void
2875 0             -22 12     u=270 imp:n=1  $ void around clad
2876 4 -2.7338  -20        u=270 imp:n=1  $ 1in graphite
2877 4 -2.7338  -21        u=270 imp:n=1  $ 2in graphite
2878 4 -2.7338  -30        u=270 imp:n=1  $ left rail
2879 4 -2.7338  -31        u=270 imp:n=1  $ right rail
2880 0            -23        u=270 imp:n=1  $ top gap
C  === fuel universe 271 slot (11, 8, 3) ===
2881 1 -19.2433  -10        u=271 imp:n=1  $ right foil
2882 2 -19.1216  -11        u=271 imp:n=1  $ left foil
2883 3 -7.162  -13        u=271 imp:n=1  $ bottom clad
2884 3 -7.162  -14        u=271 imp:n=1  $ top clad
2885 0             -15 10 11  u=271 imp:n=1  $ foil-plane void
2886 0             -22 12     u=271 imp:n=1  $ void around clad
2887 4 -2.7338  -20        u=271 imp:n=1  $ 1in graphite
2888 4 -2.7338  -21        u=271 imp:n=1  $ 2in graphite
2889 4 -2.7338  -30        u=271 imp:n=1  $ left rail
2890 4 -2.7338  -31        u=271 imp:n=1  $ right rail
2891 0            -23        u=271 imp:n=1  $ top gap
C  === fuel universe 272 slot (11, 1, 4) ===
2892 1 -18.2883  -10        u=272 imp:n=1  $ right foil
2893 2 -18.2122  -11        u=272 imp:n=1  $ left foil
2894 3 -7.2399  -13        u=272 imp:n=1  $ bottom clad
2895 3 -7.2399  -14        u=272 imp:n=1  $ top clad
2896 0             -15 10 11  u=272 imp:n=1  $ foil-plane void
2897 0             -22 12     u=272 imp:n=1  $ void around clad
2898 4 -2.7338  -20        u=272 imp:n=1  $ 1in graphite
2899 4 -2.7338  -21        u=272 imp:n=1  $ 2in graphite
2900 4 -2.7338  -30        u=272 imp:n=1  $ left rail
2901 4 -2.7338  -31        u=272 imp:n=1  $ right rail
2902 0            -23        u=272 imp:n=1  $ top gap
C  === fuel universe 273 slot (11, 3, 4) ===
2903 1 -18.3339  -10        u=273 imp:n=1  $ right foil
2904 2 -18.4555  -11        u=273 imp:n=1  $ left foil
2905 3 -7.2434  -13        u=273 imp:n=1  $ bottom clad
2906 3 -7.2434  -14        u=273 imp:n=1  $ top clad
2907 0             -15 10 11  u=273 imp:n=1  $ foil-plane void
2908 0             -22 12     u=273 imp:n=1  $ void around clad
2909 4 -2.7338  -20        u=273 imp:n=1  $ 1in graphite
2910 4 -2.7338  -21        u=273 imp:n=1  $ 2in graphite
2911 4 -2.7338  -30        u=273 imp:n=1  $ left rail
2912 4 -2.7338  -31        u=273 imp:n=1  $ right rail
2913 0            -23        u=273 imp:n=1  $ top gap
C  === fuel universe 274 slot (11, 5, 4) ===
2914 1 -18.3947  -10        u=274 imp:n=1  $ right foil
2915 2 -18.4099  -11        u=274 imp:n=1  $ left foil
2916 3 -7.171  -13        u=274 imp:n=1  $ bottom clad
2917 3 -7.171  -14        u=274 imp:n=1  $ top clad
2918 0             -15 10 11  u=274 imp:n=1  $ foil-plane void
2919 0             -22 12     u=274 imp:n=1  $ void around clad
2920 4 -2.7338  -20        u=274 imp:n=1  $ 1in graphite
2921 4 -2.7338  -21        u=274 imp:n=1  $ 2in graphite
2922 4 -2.7338  -30        u=274 imp:n=1  $ left rail
2923 4 -2.7338  -31        u=274 imp:n=1  $ right rail
2924 0            -23        u=274 imp:n=1  $ top gap
C  === fuel universe 275 slot (11, 7, 4) ===
2925 1 -18.2883  -10        u=275 imp:n=1  $ right foil
2926 2 -18.5012  -11        u=275 imp:n=1  $ left foil
2927 3 -7.4503  -13        u=275 imp:n=1  $ bottom clad
2928 3 -7.4503  -14        u=275 imp:n=1  $ top clad
2929 0             -15 10 11  u=275 imp:n=1  $ foil-plane void
2930 0             -22 12     u=275 imp:n=1  $ void around clad
2931 4 -2.7338  -20        u=275 imp:n=1  $ 1in graphite
2932 4 -2.7338  -21        u=275 imp:n=1  $ 2in graphite
2933 4 -2.7338  -30        u=275 imp:n=1  $ left rail
2934 4 -2.7338  -31        u=275 imp:n=1  $ right rail
2935 0            -23        u=275 imp:n=1  $ top gap

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
98 rpp -15.22984 228.44760 -30.45968 213.21776 -2.44348 81.37652  $ lattice container
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
     24.90343 0.00000 45.72000
     36.01593 0.00000 45.72000
     85.82279 0.00000 45.72000
     96.93529 0.00000 45.72000
     146.74215 0.00000 45.72000
     157.85465 0.00000 45.72000
     207.66151 0.00000 45.72000
     218.77401 0.00000 45.72000
     -5.55625 60.91936 45.72000
     5.55625 60.91936 45.72000
     55.36311 60.91936 45.72000
     66.47561 60.91936 45.72000
     116.28247 60.91936 45.72000
     127.39497 60.91936 45.72000
     177.20183 60.91936 45.72000
     188.31433 60.91936 45.72000
     24.90343 121.83872 45.72000
     36.01593 121.83872 45.72000
     85.82279 121.83872 45.72000
     96.93529 121.83872 45.72000
     146.74215 121.83872 45.72000
     157.85465 121.83872 45.72000
     207.66151 121.83872 45.72000
     218.77401 121.83872 45.72000
     -5.55625 182.75808 45.72000
     5.55625 182.75808 45.72000
     55.36311 182.75808 45.72000
     66.47561 182.75808 45.72000
     116.28247 182.75808 45.72000
     127.39497 182.75808 45.72000
     177.20183 182.75808 45.72000
     188.31433 182.75808 45.72000
     -5.55625 0.00000 53.34000
     5.55625 0.00000 53.34000
     55.36311 0.00000 53.34000
     66.47561 0.00000 53.34000
     116.28247 0.00000 53.34000
     127.39497 0.00000 53.34000
     177.20183 0.00000 53.34000
     188.31433 0.00000 53.34000
     24.90343 60.91936 53.34000
     36.01593 60.91936 53.34000
     85.82279 60.91936 53.34000
     96.93529 60.91936 53.34000
     146.74215 60.91936 53.34000
     157.85465 60.91936 53.34000
     207.66151 60.91936 53.34000
     218.77401 60.91936 53.34000
     -5.55625 121.83872 53.34000
     5.55625 121.83872 53.34000
     55.36311 121.83872 53.34000
     66.47561 121.83872 53.34000
     116.28247 121.83872 53.34000
     127.39497 121.83872 53.34000
     177.20183 121.83872 53.34000
     188.31433 121.83872 53.34000
     24.90343 182.75808 53.34000
     36.01593 182.75808 53.34000
     85.82279 182.75808 53.34000
     96.93529 182.75808 53.34000
     146.74215 182.75808 53.34000
     157.85465 182.75808 53.34000
     207.66151 182.75808 53.34000
     218.77401 182.75808 53.34000
     24.90343 0.00000 60.96000
     36.01593 0.00000 60.96000
     85.82279 0.00000 60.96000
     96.93529 0.00000 60.96000
     146.74215 0.00000 60.96000
     157.85465 0.00000 60.96000
     207.66151 0.00000 60.96000
     218.77401 0.00000 60.96000
     -5.55625 60.91936 60.96000
     5.55625 60.91936 60.96000
     55.36311 60.91936 60.96000
     66.47561 60.91936 60.96000
     116.28247 60.91936 60.96000
     127.39497 60.91936 60.96000
     177.20183 60.91936 60.96000
     188.31433 60.91936 60.96000
     24.90343 121.83872 60.96000
     36.01593 121.83872 60.96000
     85.82279 121.83872 60.96000
     96.93529 121.83872 60.96000
     146.74215 121.83872 60.96000
     157.85465 121.83872 60.96000
     207.66151 121.83872 60.96000
     218.77401 121.83872 60.96000
     -5.55625 182.75808 60.96000
     5.55625 182.75808 60.96000
     55.36311 182.75808 60.96000
     66.47561 182.75808 60.96000
     116.28247 182.75808 60.96000
     127.39497 182.75808 60.96000
     177.20183 182.75808 60.96000
     188.31433 182.75808 60.96000
     -5.55625 0.00000 68.58000
     5.55625 0.00000 68.58000
     55.36311 0.00000 68.58000
     66.47561 0.00000 68.58000
     116.28247 0.00000 68.58000
     127.39497 0.00000 68.58000
     177.20183 0.00000 68.58000
     188.31433 0.00000 68.58000
     24.90343 60.91936 68.58000
     36.01593 60.91936 68.58000
     85.82279 60.91936 68.58000
     96.93529 60.91936 68.58000
     146.74215 60.91936 68.58000
     157.85465 60.91936 68.58000
     207.66151 60.91936 68.58000
     218.77401 60.91936 68.58000
     -5.55625 121.83872 68.58000
     5.55625 121.83872 68.58000
     55.36311 121.83872 68.58000
     66.47561 121.83872 68.58000
     116.28247 121.83872 68.58000
     127.39497 121.83872 68.58000
     177.20183 121.83872 68.58000
     188.31433 121.83872 68.58000
     24.90343 182.75808 68.58000
     36.01593 182.75808 68.58000
     85.82279 182.75808 68.58000
     96.93529 182.75808 68.58000
     146.74215 182.75808 68.58000
     157.85465 182.75808 68.58000
     207.66151 182.75808 68.58000
     218.77401 182.75808 68.58000
     24.90343 0.00000 76.20000
     36.01593 0.00000 76.20000
     85.82279 0.00000 76.20000
     96.93529 0.00000 76.20000
     146.74215 0.00000 76.20000
     157.85465 0.00000 76.20000
     207.66151 0.00000 76.20000
     218.77401 0.00000 76.20000
     -5.55625 60.91936 76.20000
     5.55625 60.91936 76.20000
     55.36311 60.91936 76.20000
     66.47561 60.91936 76.20000
     116.28247 60.91936 76.20000
     127.39497 60.91936 76.20000
     177.20183 60.91936 76.20000
     188.31433 60.91936 76.20000
     24.90343 121.83872 76.20000
     36.01593 121.83872 76.20000
     85.82279 121.83872 76.20000
     96.93529 121.83872 76.20000
     146.74215 121.83872 76.20000
     157.85465 121.83872 76.20000
     207.66151 121.83872 76.20000
     218.77401 121.83872 76.20000
     -5.55625 182.75808 76.20000
     5.55625 182.75808 76.20000
     55.36311 182.75808 76.20000
     66.47561 182.75808 76.20000
     116.28247 182.75808 76.20000
     127.39497 182.75808 76.20000
     177.20183 182.75808 76.20000
     188.31433 182.75808 76.20000
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
