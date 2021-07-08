#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importaciones
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler, QuantileDiscretizer, OneHotEncoderEstimator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from multiprocessing.pool import ThreadPool
from IPython.display import clear_output
import time
import datetime


# In[ ]:


def init_spark():
    spark = SparkSession.builder        .master('yarn')        .config("spark.driver.memory", "8g")        .config("spark.executor.memory", "4g")        .config("spark.yarn.executor.memoryOverhead","8g")        .config("spark.driver.memoryOverhead","8g")        .config("spark.sql.crossJoin.enabled","true")        .appName("CatAutMas_VF_tg")        .getOrCreate()
    return spark


# In[ ]:


ini_time = time.time()

spark = init_spark()

exec_time = time.time() - ini_time
print("Tiempo de Inicio de Spark: " + str(datetime.timedelta(seconds=exec_time)) )


# In[ ]:


appid = spark._sc.applicationId
sparkui = "http://schlboclomp0003.corporativo.cl.corp:8088/proxy/" + appid
#print(sparkui)


# In[ ]:


#Definimos una función para realizar el proceso de precategorización
def multiDiscretizer(i, Buckets, data, varDep):
    #generamos el escenario univariable (i) para la data de entrada para valores no nulos
    
    dataVar = data.select(varDep, i).filter(i+ " IS NOT NULL AND "+i+" > " + umbral )
    #configuramos el QuantileDiscretizer
    discretizer = QuantileDiscretizer(numBuckets=Buckets, inputCol=i, outputCol='Bin')
    
    try:
        #Entrenamiento y aplicación del modelo
        result = discretizer.fit(dataVar)
        prebin = result.transform(dataVar)
        
        min1 = prebin.select(min("Bin")).first()
        if(min1[0] == 0.0):
            prebin = prebin.withColumn("Bin",prebin.Bin+1)
        
        return prebin
    except:
        #Imprimimos si hubo una variable con problemas de ejecución.
        print('Problemas con la variable: ',i)


# In[ ]:


#Definimos la función
def LimitCateg(p):
    if type(p)==pyspark.sql.dataframe.DataFrame:
        try:
            #-------------------- Creación de límites de Bins ------------------#
            #Extraemos el nombre de la variable de la columna 1
            variable = p.schema.names[1]
            #---- Contrucción de LB, Ub y totales por Bin -----------------------------------------------------#
            count_bin_max = p.groupBy("Bin").agg(max(variable).alias("Ub"))
            count_bin_min = p.groupBy("Bin").agg(min(variable).alias("Lb"))
            limitBin = count_bin_min.join(count_bin_max,["Bin"],"full")
            #Se realiza un conteo de observaciones por bin
            Count = p.groupBy(f.col('Bin')).agg(f.count('Bin').alias("TOTALES"))
            limitCount = limitBin.join(Count,["Bin"],"full").orderBy("Bin")
            return limitCount
        except:
            print('Problemas con la variable: ', variable)
            #return p
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(p), p)
        #return p


# In[ ]:


#Definimos una función para realizar el proceso de precategorización
def MultiClasificacion(p):
    #generamos el escenario univariable (i) para la data de entrada
    if type(p)==pyspark.sql.dataframe.DataFrame:
        try:
            #-------------------- Recategorizacion ------------------#
            #Extraemos el nombre de la variable de la columna 1
            variable = p.schema.names[1]
            #Borramos la columna que contiene las observaciones de la variable
            p = p.drop(variable)
            #Asginamos el nombre de la variable a la columna de bins
            clasif = p.withColumnRenamed("Bin", variable)
            return clasif
        except:
            print('Problemas con la variable: ', p.schema.names)
            #return p
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(p), p,)
        #return p 


# In[ ]:


def multiDecisionTree(t):
    
    #Arbol para Variables Numéricas Continuas:
    if type(t)==pyspark.sql.dataframe.DataFrame:
        varInd = t.schema.names[1]
        try:
            
            dt = DecisionTreeClassifier(labelCol=varDep, 
                                        featuresCol="features", 
                                        impurity='entropy',
                                        maxDepth=3, 
                                        maxBins=20, 
                                        minInstancesPerNode=20, #Número mínimo de elementos por bin
                                        #minInstancesPerNode=int(maxObs*0.05), 
                                        minInfoGain=0.0)

            varInd = t.schema.names[1]
            
            #-------------------------------------------------
            ## Transformaciones de la data para el arbol
            #-------------------------------------------------
            vector_assembler = VectorAssembler(inputCols=[varInd],outputCol="features")
            pipeline= Pipeline(stages=[vector_assembler, dt])
            model = pipeline.fit(t)
            predictions = model.transform(t)

            ##------------------------------------------------
            ##Armado de tabla de clasificaciones
            #-------------------------------------------------
            tablaBruta = predictions.select(varInd, "features", "rawPrediction", "prediction").distinct().orderBy("features")
            w = Window.orderBy(tablaBruta.rawPrediction)
            Segmentacion_label = tablaBruta.withColumn("BIN_Random_"+varInd, dense_rank().over(w))
            
            salida_df = Segmentacion_label.select(varInd, "rawPrediction", "BIN_Random_"+varInd)
            this_df = salida_df.select(salida_df[varInd], salida_df["BIN_Random_"+varInd]).groupBy("BIN_Random_"+varInd).agg(max(salida_df[varInd]).alias(varInd))
            final_df = this_df.withColumn("BIN_SANO", f.row_number().over(Window.partitionBy().orderBy(varInd))).select("BIN_SANO", varInd, "BIN_Random_"+varInd)
            
            tablaBruta = salida_df.join(final_df, salida_df["BIN_Random_"+varInd] == final_df["BIN_Random_"+varInd], how = "inner")                                  .select(final_df["BIN_SANO"].alias("BIN_"+varInd), salida_df[varInd])
            salida =t.join(tablaBruta, on = [varInd], how = 'left')
            salida = salida.drop("features", "rawPrediction")
            
            return salida

        except Exception as e:
            
            print('Problemas con la variable: ', varInd)
            print('ERROR: ', e)
            #return t
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(t), t)
        #return t


# In[ ]:


#
## DECLARACION DE FUNCION DE ARBOL DE CLASIFICACION PARA VARIABLES CATEGORICAS
def multiDecisionTreeCateg(t):
    
    if type(t)==pyspark.sql.dataframe.DataFrame:
        try:    
            dt = DecisionTreeClassifier(labelCol=varDep, 
                                        featuresCol="features", 
                                        #impurity='entropy',
                                        maxDepth=3, 
                                        maxBins=20, 
                                        minInstancesPerNode=20, #Número mínimo de elementos por bin
                                        #minInstancesPerNode=int(maxObs*0.05), 
                                        minInfoGain=0.0)

            varInd = t.schema.names[1]
            
            #-------------------------------------------------
            ## Transformaciones de la data para el arbol
            #-------------------------------------------------
            indexador = StringIndexer(inputCol=varInd, outputCol="variables", stringOrderType="alphabetAsc")
            encoder = OneHotEncoderEstimator(inputCols=["variables"],outputCols=["features"], dropLast=False)
            
            pipeline= Pipeline(stages=[indexador, encoder, dt])
            
            model = pipeline.fit(t)
            predictions = model.transform(t)

            ##------------------------------------------------
            ##Armado de tabla de clasificaciones
            #-------------------------------------------------
            
            tablaBruta = predictions.select(varInd, "features", "rawPrediction", "prediction").distinct().orderBy("features")
            w = Window.orderBy(tablaBruta.rawPrediction)
            Segmentacion_label = tablaBruta.withColumn("BIN_Random_"+varInd, dense_rank().over(w))
            
            salida_df = Segmentacion_label.select(varInd, "rawPrediction", "BIN_Random_"+varInd)
            this_df = salida_df.select(salida_df[varInd], salida_df["BIN_Random_"+varInd]).groupBy("BIN_Random_"+varInd).agg(max(salida_df[varInd]).alias(varInd))
            final_df = this_df.withColumn("BIN_SANO", f.row_number().over(Window.partitionBy().orderBy(varInd))).select("BIN_SANO", varInd, "BIN_Random_"+varInd)
            
            tablaBruta = salida_df.join(final_df, salida_df["BIN_Random_"+varInd] == final_df["BIN_Random_"+varInd], how = "inner")                                  .select(final_df["BIN_SANO"].alias("BIN_"+varInd), salida_df[varInd])
            salida =t.join(tablaBruta, on = [varInd], how = 'left')
            salida = salida.drop("features", "rawPrediction")
            
            return salida

        except BaseException as e:
            print('Problemas con la variable: ', varInd)
            print('ERROR: ', e)
            #return t
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(t), t)
        #return t


# In[ ]:


def LimitCategBinning(p):
    if type(p)==pyspark.sql.dataframe.DataFrame:
        try:
            #-------------------- Creación de límites de Bins ------------------#
            #Extraemos el nombre de la variable de la columna 1
            varInd = p.schema.names[0]
            #---- Contrucción de LB, Ub y totales por Bin -----------------------------------------------------#
            #---- Contrucción de LB, Ub y totales por Bin -----------------------------------------------------#
            count_bin = p.groupBy("Bin_"+varInd).agg(min(varInd).alias("Lb"), max(varInd)                                                .alias("Ub"),f.count('Bin_'+varInd).alias("TOTALES"))                                                .sort('Bin_'+varInd)
            
            return count_bin
        except:
            print('Problemas con la variable: ', varInd)
            #return p
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(p), p)
        #return p 


# In[ ]:


def limitesOriginales(l, limitCateg):
    if type(l)==pyspark.sql.dataframe.DataFrame:
        try:
            #-------- Extraccion de Variable--------------#
            varInd = l.schema.names[0].split("Bin_")[1]
            index = dfVarsNum.schema.names.index(varInd)
            q = limitCateg[index]
            #------- Proceso de Joins ---------------------#
            #Cambio de nombres de columnas
            LimBin=l.withColumnRenamed("Lb","Lb_1")                    .withColumnRenamed("Ub","Ub_1")                    .withColumnRenamed("TOTALES","TOTALES_1")
            LimBin = LimBin.join(q,LimBin.Lb_1==q.Bin,"left")
            LimBin = LimBin.withColumnRenamed("Lb","Lb_2")                           .withColumnRenamed("Ub","Ub_2")                           .withColumnRenamed("TOTALES","TOTALES_2")                           .withColumnRenamed("Bin","Bin_1")
            LimBin =  LimBin.join(q, LimBin.Ub_1==q.Bin, "left")
            LimBin=LimBin.drop("Lb_1").drop("Ub_1").drop("Bin_1")                         .drop("Ub_2").drop("TOTALES_2")                         .drop("Lb").drop("Bin")                         .drop("TOTALES")
            LimBin=LimBin.withColumnRenamed("Lb_2","Lb")                         .withColumnRenamed("TOTALES_1","TOTALES").orderBy("Bin_"+varInd)
            return LimBin
        except Exception as e:
            print('Problemas con la variable: ', varInd)
            print('ERROR: ', e)
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(l), l)
        #return l  


# In[ ]:



def MultiAsignacionCategorias(p, LimitesOriginales, dfVarsNum):   
    
    cotas = 1
    if type(p)==pyspark.sql.dataframe.DataFrame:
        try:
            #-------- Extraccion de Variable--------------#
            varInd = p.schema.names[1]
            index = dfVarsNum.schema.names.index(varInd)
            q = LimitesOriginales[index]
            binvarInd = "Bin_"+varInd  
            if(cotas != 1):                           
                df_final = p.join(q, (((p[varInd] < q["Ub"]) & (p[varInd] >= q["Lb"])) | ((p[varInd] == q["Ub"]) & (p[varInd] == q["Lb"]))), how = "left")                                .select(p["*"], q[binvarInd].alias("PRE_" + varInd))
            else:
                df_final = p.join(q, (((p[varInd] < q["Ub"]) & (p[varInd] >= q["Lb"])) | ((p[varInd] == q["Ub"]) & (p[varInd] == q["Lb"]))), how = "left")                                .select(p["*"], q[binvarInd].alias("PRE_" + varInd), q["Lb"], q["Ub"])          
            return df_final
        except Exception as e:
            print('Problemas con la variable: ', varInd)
            print('ERROR: ', e)
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(p), p)
        #return p


# In[ ]:


def MultiIvWoe(n, data_l):
    
    if type(n)==pyspark.sql.dataframe.DataFrame:
        try:
            varInd = n.schema.names[1]
            pre_var ="PRE_"+varInd
            var_lb = "lb"
            var_ub = "ub"
            #---------------------------------------------------------------------------------------------------------------------------#
            newdataX = n.select(varInd, pre_var, var_lb, var_ub, varDep)                        .groupBy(pre_var, var_lb, var_ub)                        .agg(count(varDep).alias("count"), (count(n[varDep])-sum(n[varDep])).alias("good"), sum(n[varDep]).alias("bad"))                        .withColumnRenamed(pre_var, "bin").withColumn("variable", lit(varInd))
            
            ############ 
            df_flags = data_l.select(varDep, varInd).where(varInd + " <= " + umbral).agg(count(varInd).alias("count"),                                                                         (count(data_l[varDep])-sum(data_l[varDep])).alias("good"),                                                                         sum(data_l[varDep]).alias("bad"), min(varInd).alias("lb"),                                                                         max(varInd).alias("ub"))                                                                        .withColumn("variable", lit(varInd))
            if(df_flags.count() > 0):
                df_flags = df_flags.withColumn("bin", lit('NA')).select("bin","lb", "ub", "count", "good", "bad", "variable") 
                newdataX = newdataX.union(df_flags) 
            
            ########


            newdataX2 = newdataX.select("variable", "good", "bad").groupBy("variable")                            .agg(sum(newdataX["good"]).alias("total_buenos"), sum(newdataX["bad"]).alias("total_malos"))

            postdata = newdataX.join(newdataX2, newdataX["variable"] == newdataX2["variable"], how = 'inner')                                .select(newdataX["variable"], newdataX["bin"], newdataX["lb"], newdataX["ub"],                                 newdataX["count"], newdataX["good"], newdataX["bad"], (newdataX["good"]/newdataX2["total_buenos"]).alias("good_distr"),                                 (newdataX["bad"]/newdataX2["total_malos"]).alias("bad_distr"), (newdataX["bad"]/newdataX["count"]).alias("badprob"))

            postdata = postdata.withColumn("woe", log((postdata.good_distr/postdata.bad_distr)))                               .withColumn("iv", (postdata.good_distr-postdata.bad_distr)*log(postdata.good_distr/postdata.bad_distr))

            sumas = postdata.select("variable", "iv").groupBy("variable").agg(sum(postdata["iv"]).alias("iv"))

            df3 = postdata.join(sumas, postdata["variable"] == sumas["variable"], how = "inner")                            .select(postdata["variable"], postdata["bin"], postdata["lb"], postdata["ub"], postdata["count"],                             postdata["good"], postdata["bad"], round(postdata["good_distr"], 5).alias("good_distr"), round(postdata["bad_distr"], 5).alias("bad_distr"), round(postdata["badprob"], 5).alias("badprob"),                            round(postdata["woe"], 5).alias("woe"), round(postdata["iv"], 5).alias("iv"), round(sumas["iv"], 5).alias("iv_total"))
            
            #max1 = df3.select(max("ub")).first()
            #min1 = df3.select(min("lb")).first()
            #df3 = df3.withColumn("ub", when(df3["ub"] == max1[0],  lit('Infinity') ).otherwise(df3["ub"]))\
            #            .withColumn("lb", when(df3["lb"] == min1[0], lit('-Infinity') ).otherwise(df3["lb"]))
            
            return df3
        except BaseException as e:
            print('Problemas con la variable: ', varInd)
            print('ERROR: ', e)
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(n), n)
        #return n
        


# In[ ]:


#Se define ensamblador de lista de data frames de variables categoricas
def ensambladorC(i, dataC, varDep):
    dataVar = dataC.select(varDep, i).filter(i+ " IS NOT NULL")
    return dataVar


# In[ ]:


def MultiIvWoeCategorico(n):
    
    if type(n)==pyspark.sql.dataframe.DataFrame:
        try:
            varInd = n.schema.names[0]
            pre_var ="BIN_"+varInd
            #---------------------------------------------------------------------------------------------------------------------------#
            newdataX = n.select(varInd, varDep, pre_var)                        .groupBy(pre_var)                        .agg(count(varDep).alias("count"), (count(n[varDep])-sum(n[varDep])).alias("good"), sum(n[varDep]).alias("bad"))                        .withColumnRenamed(pre_var, "bin").withColumn("variable", lit(varInd))
            
            ##-------------->
            
            newdataX2 = newdataX.select("variable", "good", "bad").groupBy("variable")                            .agg(sum(newdataX["good"]).alias("total_buenos"), sum(newdataX["bad"]).alias("total_malos"))
            
            postdata = newdataX.join(newdataX2, newdataX["variable"] == newdataX2["variable"], how = 'inner')                                .select(newdataX["variable"], newdataX["bin"],                                 newdataX["count"], newdataX["good"], newdataX["bad"], (newdataX["good"]/newdataX2["total_buenos"]).alias("good_distr"),                                 (newdataX["bad"]/newdataX2["total_malos"]).alias("bad_distr"), (newdataX["bad"]/newdataX["count"]).alias("badprob"))

            postdata = postdata.withColumn("woe", log((postdata.good_distr/postdata.bad_distr)))                               .withColumn("iv", (postdata.good_distr-postdata.bad_distr)*log(postdata.good_distr/postdata.bad_distr))

            sumas = postdata.select("variable", "iv").groupBy("variable").agg(sum(postdata["iv"]).alias("iv"))

            df3 = postdata.join(sumas, postdata["variable"] == sumas["variable"], how = "inner")                            .select(postdata["variable"], postdata["bin"], postdata["count"],                             postdata["good"], postdata["bad"], round(postdata["good_distr"], 5).alias("good_distr"), round(postdata["bad_distr"], 5).alias("bad_distr"), round(postdata["badprob"], 5).alias("badprob"),                            round(postdata["woe"], 5).alias("woe"), round(postdata["iv"], 5).alias("iv"), round(sumas["iv"], 5).alias("iv_total"))
            df3 = df3.withColumn("lb", lit("NA")).withColumn("ub", lit("NA"))
            
            return df3
        except BaseException as e:
            print('Problemas con la variable: ', varInd)
            print('ERROR: ', e)
    else:
        print('######## VARIABLE NO PROCESADA: ######## ', type(n), n)
        #return n
        


# In[ ]:


##################################################################################
####                       -> CONFIGURACION <-                                ####
##################################################################################
######################## DATOS A USAR ############################################
##################################################################################

#########Importamos la data que contiene las variables a utilizar################
#
#df_master = spark.table("greporting_riesgos.mb_fuga_rrhh_vars_iv")
df_master = spark.table("grsg_modelos.data_ok")
#DictCateg = spark.table("grsg_modelos.aci_fuga")
DictCateg = spark.table("greporting_riesgos.dicc_data_ok")


### ---> Caso especial  -> data_ok
DictCateg = DictCateg.drop("IND_BM")
DictCateg = DictCateg.drop("ID")
### ----->


#Variable Dependiente de estudio
varDep='ind_bm'
#varDep = 'ind_fuga_09'
#Numero de separaciones para clasificacion de Pre_binning
Buckets = 20
#Creamos un pool de 10 hilos que corren en paralelo
pool = ThreadPool(10)
#Umbral para variables especiales
#Para no utilizar valores especiales, dejar umbral como un valor extremadamente negativo. ej: 
umbral = '-777770'
#--------------------------------------------------------------------------------#
#Imprimimos la lista
aci_fuga = DictCateg.rdd.map(lambda x: x.name)
#recolectar los valores del rdd en una lista
Variables = aci_fuga.collect()
#procesar el encode de la lista
Variables = [s.encode('ascii') for s in Variables]
#print(Variables)
#--------------------------------------------------------------------------------#
#Extraemos el subset de datos utilizados para el estudio
#ASignamos el nombre de la variable dependiente
#Añadimos a la lista Variables, la variable dependiente
Variables.append(varDep)
#Creamos un DataFrame que contiene el total de variables para el estudio
dfVars = df_master.select([col for col in Variables])
#---------------------------------------------------------------------------------#
#Extraemos las variables de la columna names que tengan el valor de type = 'Char' a través de RDD
names = DictCateg.select("name").where("type_='Char'").rdd.map(lambda x: x.name)
categoricas = names.collect()
#Añadimos la variable dependiente a la lista
categoricas.append(varDep)
#Codificamos la lista en el caso de que no venga en utf-8
categoricas = [s.encode('ascii') for s in categoricas]
#Extraemos las variables de la columna names que tengan el valor de type = 'Num' a través de RDD
names = DictCateg.select("name").where("type_='Num'").rdd.map(lambda x: x.name)
numericas = names.collect()
numericas.append(varDep)
#Codificamos la lista en el caso de que no venga en utf-8
numericas = [s.encode('ascii') for s in numericas]


#########################CASO ESPECIAL DATA OK    ##################################
numericas.remove("IND_BM")#Variable dependiente se almacena en otra parte. sacar del diccionario
numericas.remove("ID")#Identificador que no entrega valor al estudio. sacar del diccionario
numericas.remove("CHAR052")#Variable mal formateada de origen
numericas.remove("ldn_pas_cta_cte_mn_sm")#Variable mal formateada de origen
####################################################################################
#########################CASO ESPECIAL FUGA     ####################################
#numericas.remove("desempenio")#Variable viene con solo NULLs
####################################################################################


#Extracción de DataFrame de variables categóricas
dfVarsCateg = dfVars.select([col for col in categoricas])
#Extracción de DataFrame de variables Numéricas
dfVarsNum = dfVars.select([col for col  in numericas])

#Creamos un lista de parámetros que contiene la lista de variables excepto la variable dependiente
parameters = numericas[:len(numericas)-2]
parametersC = categoricas[:len(categoricas)-2]
#DAta NUMERICA final a usar:
data = dfVarsNum
#DAta CATEGORICA final a usar:
dataC = dfVarsCateg


# In[ ]:


################################################################################
################             EJECUCIONES             ###########################
################################################################################


#
#Se ejecuta prebinning para variables Numericas
ini = time.time()
preBinning = pool.map(lambda i: multiDiscretizer(i, Buckets, data, varDep), parameters)
#total = time() - ini
#print('Duracion Discretizer: ', total,'Segundos')
#print("\n")

#
#Se ejecuta funcion de asignacion de categorias obtenidas en prebinning a la data de entrada
#ini00 = time()
limitCateg = pool.map(lambda p:  LimitCateg(p), preBinning)
#total = time() - ini00
#print('Duracion limit Categ: ', total,' Segundos')
#print("\n")

#
#Se ejecuta funcion de creacion de columna bins con clasificacion obtenida
#ini0 = time()
clasificacion = pool.map(lambda p:  MultiClasificacion(p), preBinning)
#total0 = time() - ini0
#print('Duracion multi clasificacion: ', total0,' Segundos')
#print("\n")

#
#Se ejecuta funcion de Binning de clasificacion con arbol de decicion sobre resultados del prebinning
#ini1 = time()
Binning = pool.map(lambda t:  multiDecisionTree(t), clasificacion)
#total1 = time() - ini1
#print('Duracion Arbol de Decision: ', total1/60,' Minutos')
#print("\n")

#
#Se ejecuta funcion que asigna la nueva clasificacion obtenida desde el arbol a la data de entrada
#ini2 = time()
limitBinning = pool.map(lambda p:  LimitCategBinning(p), Binning)
#total2 = time() - ini2
#print('Duracion LimitCategBinning: ', total2,' Segundos')
#print("\n")

#
#Se ejecuta funcion que devuelve los limites originales al resultado del binning
#ini5 = time()
LimitesOriginales = pool.map(lambda l: limitesOriginales(l, limitCateg), limitBinning)
#total5 = time() - ini5
#print('Duracion limites originales: ', total5,' Segundos')
#print("\n")

#
#Se ejecuta funcion que asigna la clasificacion y limites originales a la data de entrada
#ini3 = time()
New_Cat = pool.map(lambda p: MultiAsignacionCategorias(p, LimitesOriginales, dfVarsNum), preBinning)
#total3 = time() - ini3
#print('Duracion MultiAsignacionCategorias: ', total3,' Segundos')
#print("\n")

#
#Se ejecuta la funcion que calcula la tabla IV WOE de cada variable Numerica
#ini4 = time()
IVWOE_Master = pool.map(lambda n: MultiIvWoe(n, data), New_Cat)
#total4 = time() - ini4
#print('Duracion creacion de tablas IV WOE: ', total4/60,' Minutos')
#print("\n")
#################################################################
### Fin analisis de variables numericas                         #
#################################################################
#
#
#Se ejecuta la funcion que genera lista de variables categoricas
#inic = time()
ensambladorCat = pool.map(lambda e: ensambladorC(e, dataC, varDep), parametersC)
BinningCateg = pool.map(lambda t:  multiDecisionTreeCateg(t), ensambladorCat)
IVWOE_CAT = pool.map(lambda p:  MultiIvWoeCategorico(p), BinningCateg)
#totalc = time() - inic
#print('Duracion creacion de tablas IV WOE categorico: ', totalc,' Segundos')
#print("\n")

#################################################################
### Fin analisis de variables categoricas                       #
#################################################################

totalfinal = time.time() - ini
print('Duracion Total categorizacion_automatica: ', totalfinal/60 ,' Minutos')

print("\n")


# #### Escritura en Datalake

# In[ ]:


ini = time.time()
##Limpieza de la tabla que estaba ingestada antes
###
vacio = IVWOE_Master[0].limit(0)
vacio.write.mode("overwrite").saveAsTable("grsg_modelos.IV_WOE_MAESTRA_" + varDep)
i=0
ivwoe_completa = IVWOE_Master + IVWOE_CAT
for x in ivwoe_completa:    
    x.write.mode("append").saveAsTable("grsg_modelos.IV_WOE_MAESTRA_" + varDep)
    i = i+1
    clear_output(wait=True)
    print("Se han escrito --->  " + str(i)+" variables")
total = time.time()-ini
print('Tiempo de migracion tabla IV WOE Maestra ', total/60,' Minutos')


# In[ ]:


spark.stop()


# In[ ]:




