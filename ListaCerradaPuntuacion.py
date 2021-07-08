#######################################################################################################################
#                                                                                                                     #
#                                                   HDU225                                                            #
#                                             Lista Cerrada Puntuación                                                #
#                                                                                                                     #
#El código se divide en los siguientes pasos:                                                                         #
#                                                                                                                     #
#   a) Creación  y tratamiento de LISTA CERRADA usando como input                                                     #
#   b) RISKMI.LISTA_CERRADA_&MES.                                                                                     #
#   c) RISKMI.ID_CLIENTE_NEW                                                                                          #
#   d) Se usa código de corporativo, haciendo match con el local*                                                     #
#   e) Creación de DATA de clientes                                                                                   #
#   f) JOIN entre data clientes y Lista Cerrada y tratamiento de tablas                                               #
#   g) Exportar archivo V2_LISTA_CERRADA_AAAAMM_&MM.txt                                                               #
#######################################################################################################################
# Tiempo de ejecucion probado: 02:02                                                                                  #
#######################################################################################################################


#######################################################################################################################
###################################### IMPORTACIONES E INICIALIZACION DEL SPARK CONTEXT ###############################
#######################################################################################################################

# Bibliotecas
#-------------------------------------------#
import time
import operator
import datetime
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
#----------------------------------------------------------------#
def init_spark():
  spark = spark = SparkSession.builder\
        .master('yarn')\
        .appName("hdu225")\
        .getOrCreate()
  sc = spark.sparkContext
  sqlContext = SQLContext(sc)
  return spark,sc,sqlContext

spark, sc, sqlContext = init_spark()
#----------------------------------------------------------------#

#################################
## Configuracion de Parametros ##
#################################

mes_actual = '202102'

fecha = '31JAN2021:00:00:00'

fecha_proceso = '28-02-2021'

mes_anterior = '202101'

#######################
## Tablas Input #######
#######################
#Esta tabla es proveniente del area de Reporting y Consolidación
#Su Workspace es greporting_riesgos
####################################################################

bdd_lista_cerrada_mes = "gsisteinforiesgo.lista_cerrada_"+mes_actual

bdd_id_cliente_new = "gsisteinforiesgo.id_cliente_new"

bdd_jm_client_bii = "gsisteinforiesgo.jm_client_bii"
#bdd_jm_client_bii = "bu_bdr_hist.jm_client_bii"
#bdd_jm_client_bii = "bu_bdr.jm_client_bii"

bdd_dt_secuencial_cliente = "gsisteinforiesgo.bdr_dt_secuencial_cliente"

bdd_lista_cerrada_mes_antes = "gsisteinforiesgo.lista_cerrada_201012_"+mes_anterior

#############################
###### Paso 1 ###### Paso 1 #
# Creacion y tratamiento de #
# Lista Cerrada             #
#############################
#Se comienza a medir el tiempo de ejecucion del proceso
start_time = time.time()


#RISKMI.LISTA_CERRADA_&MES AS A

df_listaCerradaMes = spark.table(bdd_lista_cerrada_mes)
df_listaCerradaMes.persist()

#RISKMI.ID_CLIENTE_NEW AS B 

df_id_cliente_new = spark.table(bdd_id_cliente_new)
df_id_cliente_new.persist()

#Se crea tabla LIST_CERRADA_MES

df_list_cerrada_mes = df_listaCerradaMes.join\
                                         (df_id_cliente_new.select(df_id_cliente_new.NUMERO, df_id_cliente_new.IDF_PERS_KGR),\
                                         df_listaCerradaMes.Unidad_Operativa == df_id_cliente_new.IDF_PERS_KGR,\
                                         "left")

df_list_cerrada_mes.persist()

df_id_cliente_new.unpersist()
df_listaCerradaMes.unpersist()

#Crea LIST_CERRADA_MES_2 a partir de LIST_CERRADA_MES
#Se comienza a abreviar LIST_CERRADA_MES_2 como LCM2

df_LCM2 = df_list_cerrada_mes.filter(df_list_cerrada_mes.NUMERO.isNotNull() )

####################################
### SEGMENTO LOCAL y PRIORIDAD_2 ###
####################################
df_LCM2_1 = df_LCM2.withColumn("SEGMENTO_LOCAL",\
                             when( (df_LCM2.Tipo_Entidad == 'PROJECT FINANCE')\
                             & (df_LCM2.Sector_Interno_1.isin(5090, 5091, 5095) ), 150)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'CORPORATE_IR'),\
                                  130)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'BANKS_IR'),\
                                  112)\
                              .otherwise( when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                             & (df_LCM2.Sector_Interno_1.isin(1500, 1501, 1505) ) , 113)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                             & (df_LCM2.Sector_Interno_1.isin(9000, 9001, 9005) ) , 114)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno.isin('NON-BANK_IFI_IR','FONDS_IR') )\
                             & (df_LCM2.Sector_Interno_1.isin(6500, 6501, 6505, 6600, 6601, 6605) ) , 115)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                             & (df_LCM2.Sector_Interno_1.isin(4000,4001,6000,6001,6005) ) , 116)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR'),\
                                  118)\
                              .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'SOVEREIGN_IR'),\
                                  101) ) ) ) ) ) ) ) ) )\
                   .withColumn("PRIORIDAD_2",\
                             when( (df_LCM2.Tipo_Entidad == 'PROJECT FINANCE') \
                             & (df_LCM2.Sector_Interno_1.isin(5090, 5091, 5095)), 18)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'CORPORATE_IR'), 15)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'BANKS_IR'), 10)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR') \
                             & (df_LCM2.Sector_Interno_1.isin(1500, 1501, 1505) ) , 11)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR') \
                             & (df_LCM2.Sector_Interno_1.isin(9000, 9001, 9005) ) , 12)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno.isin('NON-BANK_IFI_IR','FONDS_IR')) \
                             & (df_LCM2.Sector_Interno_1.isin(6500, 6501, 6505, 6600, 6601, 6605) ) , 13)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR') \
                             & (df_LCM2.Sector_Interno_1.isin(4000,4001,6000,6001,6005) ) , 12)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'NON-BANK_IFI_IR'), 14)\
                             .otherwise(when( (df_LCM2.Modelo_Rating_Interno == 'SOVEREIGN_IR'), 1)) ) ) ) ) ) ) ) )


#Se crea list_cerrada_mes_3 a partir de list_cerrada_mes_2

df_LCM2_3 = df_LCM2_1.filter(df_LCM2_1.SEGMENTO_LOCAL.isNotNull() )

df_LCM2_3.persist()
#----------------------------------------

df_LCM2_3 = df_LCM2_3.withColumn("ID_MODEL_LC",\
                      when((df_LCM2_3.Modelo_Rating_Interno == 'BANKS_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),1001 )\
                      .otherwise( when((df_LCM2_3.Modelo_Rating_Interno == 'BANKS_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),1002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'BANKS_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),1003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'BANKS_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),1004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'BANKS_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),1005 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'CORPORATE_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),2001 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'CORPORATE_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),2002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'CORPORATE_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),2003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'CORPORATE_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),2004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'CORPORATE_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),2005 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'FONDS_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),3001 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'FONDS_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),3002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'FONDS_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),3003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'FONDS_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),3004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'FONDS_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),3005 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'INTRAGROUPS_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),4001 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'INTRAGROUPS_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),4002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'INTRAGROUPS_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),4003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'INTRAGROUPS_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),4004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'INTRAGROUPS_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),4005 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),5001 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),5002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),5003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),5004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'NON-BANK_IFI_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),5005 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'PROJECT_FINANCE_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),6001 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'PROJECT_FINANCE_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),6002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'PROJECT_FINANCE_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),6003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'PROJECT_FINANCE_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),6004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'PROJECT_FINANCE_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),6005 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'SOVEREIGN_IR')\
                         & (df_LCM2_3.Prioridad == "P1"),7001 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'SOVEREIGN_IR')\
                         & (df_LCM2_3.Prioridad == "P2"),7002 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'SOVEREIGN_IR')\
                         & (df_LCM2_3.Prioridad == "P3"),7003 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'SOVEREIGN_IR')\
                         & (df_LCM2_3.Prioridad == "P4"),7004 )\
                      .otherwise(when((df_LCM2_3.Modelo_Rating_Interno == 'SOVEREIGN_IR')\
                         & (df_LCM2_3.Prioridad.isNull()),7005 )\
                      .otherwise(99999)) )) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) ) )

df_LCM2_3.persist()
#----------------------------------------

###########################################################################################

df_LCM3 = df_LCM2_3.withColumn("TIPMODE2",\
                      when(df_LCM2_3.ID_MODEL_LC == 1004, 3 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 1005, 3 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 2001, 5 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 2002, 5 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 2003, 5 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 2004, 5 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 2005, 5 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 3005, 4 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 5004, 2 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 5005, 2 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 6005, 8 )\
                      .otherwise(when(df_LCM2_3.ID_MODEL_LC == 7005, 1 )\
                                .otherwise(99) ) ) ) ) ) ) ) ) ) ) ) )


df_LCM3.persist()
df_LCM2_3.unpersist()

#----------------------------------------------
#Se crea List Cerrada Mes 4 a partir de la 3

df_LCM4 = df_LCM3.withColumn("IDPUNSCO",\
                            when( (df_LCM3.Rating_Interno == 'SIN_CALIFI'), 999)\
                            .otherwise(df_LCM3.Rating_Interno*1) )

df_LCM4.persist()

df_LCM3.unpersist()
#########################################################
### PASO 2 ### PASO 2 ### PASO 2 ### PASO 2 ### PASO 2  #
#########################################################
df_jmClientBii = spark.table(bdd_jm_client_bii)
df_jmClientBii.persist()

df_cliente = df_jmClientBii.filter(df_jmClientBii.FEOPERAC == fecha)

#funcion scan rut
df_cliente = df_cliente.withColumn("RUT",\
                split((df_cliente.CODIDPER),"-")[0] )

df_cliente.persist()

#Crear List Cerrada Mes 5
# A = list cerrada mes 4
# B = cliente

df_LCM5 = df_LCM4.join( df_cliente.select("IDNUMCLI", "CODIDPER", "RUT"),\
                        df_LCM4.NUMERO == df_cliente.RUT, "left")

#-----------------------------------------------------------------------------------
df_LCM5.persist()
df_cliente.unpersist()
df_LCM4.unpersist()

##LPAD = FORMAT 19.1 en SAS
df_LCM5.withColumn("IDPUNSCO",\
                      round( lpad(df_LCM5.IDPUNSCO, 19, "0"), 1) )


#Crear List Cerrada Mes 6

df_LCM6 = df_LCM5.withColumn("TIPO", when(df_LCM5.IDNUMCLI.isNotNull(), 94 ) )\
                 .withColumn("FECHA", when(df_LCM5.IDNUMCLI.isNotNull(), fecha_proceso ) )\
                 .withColumn("CLISEGL1", when(df_LCM5.IDNUMCLI.isNotNull(), df_LCM5.SEGMENTO_LOCAL ) )\
                 .withColumn("APNOMPER", when(df_LCM5.IDNUMCLI.isNotNull(), df_LCM5.Nombre_Uni_Oper_KGL ) )

###########################
## Se eliminan los nulos ##
###########################
df_LCM6 = df_LCM6.filter(df_LCM6.IDNUMCLI.isNotNull())
     
         
df_LCM6.persist()
df_LCM5.unpersist()
####
#Creacion List Cerrada Mes 7
# A LIST CERRADA MES 6
# B DT SECUENCIAL CLIENTE
df_dtSecuencialCliente = spark.table(bdd_dt_secuencial_cliente)
df_dtSecuencialCliente.persist()

df_LCM7 = df_LCM6.join(df_dtSecuencialCliente.select("IDF_PERS_ODS", "COD_CLIENTE_BDR"),\
                      (df_LCM6.IDNUMCLI == df_dtSecuencialCliente.COD_CLIENTE_BDR),\
                      "left")

df_LCM7.persist()
df_LCM6.unpersist()
#############################
# LIST CERRADA MES 8        #
#############################

df_LCM8 = df_LCM7.select("IDNUMCLI",	"APNOMPER",	"CODIDPER",\
                         "CLISEGL1",	"TIPMODE2",	"TIPO" ,\
                         "IDPUNSCO",	"FECHA","IDF_PERS_ODS")

df_LCM8.persist()
df_LCM7.unpersist()

#----------------------------------------------------------------------#
#Se apilan  lista_cerrada_mes_antes y LIST_CERRADA_MES_8
df_listaCerradaMesAntes = spark.table(bdd_lista_cerrada_mes_antes)
df_listaCerradaMesAntes.persist()


df_LCM8 = df_LCM8.union(df_listaCerradaMesAntes)

df_LCM8.persist()
df_listaCerradaMesAntes.unpersist()
#----------------------------------------------------------------------#
#Se escribe en el lago

df_LCM8.write.mode("overwrite").saveAsTable("gsisteinforiesgo.lista_cerrada_201012_"+mes_actual)


exec_time = time.time() - start_time
print("Tiempo de Ejecucion: " +str(datetime.timedelta(seconds=exec_time)) )
######################
### FIN DE PROCESO ###
######################

spark.stop()

######################
### FIN DE PROCESO ###
######################