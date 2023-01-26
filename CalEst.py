#!/usr/bin/python3
# coding=utf-8*
# author mlizarazo@sgc.gov.co


import os, glob
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, HourLocator, MinuteLocator, DateFormatter
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import numpy as np
import sys
from datetime import date, datetime, timedelta
from obspy import read
from obspy import UTCDateTime
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import seaborn as sns
import pandas as pd 
import csv

sns.set(style="dark")

dir_input = sys.argv[1]
#dir_input="inp_calidad.txt"

#---- gestiona input
arc_inp=open(dir_input,"r")
lineas_inp=arc_inp.readlines()
for l in lineas_inp:
    
    if l[0]!="#": #omite comentarios en el .inp
        
        if "time_i" in l:                        
            time_i=UTCDateTime(l.split("=")[1].replace("\n","")).datetime            
        if "time_f" in l:            
            time_f=UTCDateTime(l.split("=")[1].replace("\n","")).datetime            
        if "Estacion" in l:            
            estacion_name=str(l.split("=")[1].replace("\n","").replace(" ","").upper())                        
        if "Canal" in l:            
            ch=str(l.split("=")[1].replace("\n","").replace(" ","").upper())
        if "Servidor" in l:            
            servidor_elegido=str(l.split("=")[1].replace("\n","").replace(" ",""))
            if servidor_elegido=="sc13":
                servidor_elegido="sc13/mseed"

arc_inp.close()            
#print estacion_name,ch,time_i,time_f,servidor_elegido

def encuentra_picos(data_ventana,off_v,desv_stand_v,nm_d):
    
    fun=np.abs(data_ventana-off_v)
    umbral=5
    flat=0.001
    distancia_mini_pic=50 #segunos

    ind_fun_arri=[]
    for i in range(0,len(fun)-4):
        if fun[i] > umbral*desv_stand_v and fun[i+1] > umbral*desv_stand_v and fun[i+2] > umbral*desv_stand_v and fun[i+3] > umbral*desv_stand_v and fun[i+4] > umbral*desv_stand_v:
            if (fun[i+1] - fun[i]) < 2*flat*desv_stand_v and (fun[i+2] - fun[i+1]) < flat*desv_stand_v and (fun[i+2] - fun[i+3]) < flat*desv_stand_v and (fun[i+3] - fun[i+4]) < 2*flat*desv_stand_v:
                ind_fun_arri.append(i+1)
    
    c=0
    r=0
    j=0
    i=0
    
    indices_unicos=[]
    
    while j < len(ind_fun_arri)-2 and i < len(ind_fun_arri)-2 and r < 100 :
        for i in range(c,len(ind_fun_arri)):
            fam=[]            
            for j in range(i+1,len(ind_fun_arri)):        
                if ind_fun_arri[j] - ind_fun_arri[i] < nm_d*distancia_mini_pic:
                    fam.append(ind_fun_arri[i]) #repite el mismo valor de inicio i
                else:                    
                    break #muere ciclo j

            c=j            
            if len(fam)>0:
                indices_unicos.append(fam[0])
            break #muere ciclo i

        r=r+1
        
    return indices_unicos

def obtiene_off_picks(st):    

    time_peaks_v=[]
    data_total=[]

    for tr in st:                              
            
        data=tr.data
        data_total=np.concatenate((data_total, data), axis=None)                            
           
        del_tr=tr.stats.delta            
                                                                                                           
        nm=1/del_tr
        decimacion=int(nm/20) #remuestrea a 20hz, obtiene picos a esta tasa para mejorar velocidad de computo    
        
        tr_p=tr.copy()
        tr_p.decimate(decimacion, strict_length=False,no_filter=True)
        del_tr_p=tr_p.stats.delta
        
        data_dec=tr_p.data
        
        nm_d=1/del_tr_p #numero de muestras por s en la traza decimada
        nm_v=int(nm_d*60*60*2) #numero de muestras de cada ventana, corta ventanas de maximo 2 horas, para analizar picos
        
        num_ventanas=int(len(data_dec)/nm_v)+1
        
        pos_pics_v=[]
        for i in range(1,num_ventanas+1):

            if (i-1)*nm_v < len(data_dec):
                if i*nm_v < len(data_dec):
                    data_ventana=data_dec[(i-1)*nm_v:i*nm_v]
                if i*nm_v > len(data_dec):
                    data_ventana=data_dec[(i-1)*nm_v:len(data_dec)]

                off_v=round(np.mean(data_ventana),1)
                desv_stand_v=np.std(data_ventana)
                #peaks_v, _ = find_peaks(np.abs(data_ventana-off_v), height=5*desv_stand_v,distance=nm_d*40,plateau_size=2) #platu parece no corerr en scipy python 2, el 219 es python 2
                ind_fun_arri=encuentra_picos(data_ventana,off_v,desv_stand_v,nm_d)
                       
                for pv in ind_fun_arri:
                    pos_pics_v.append(pv + ((i-1)*nm_v))
            
        time_i=tr_p.stats.starttime.datetime                                
       
        for pv in pos_pics_v:
            time_peak_v= time_i + timedelta(seconds=pv*del_tr_p)
            #print ("v",time_peak_v)
            time_peaks_v.append(time_peak_v) 
                                   
    num_picks=len(time_peaks_v)
    offs=np.mean(data_total)
    desv_stand=np.std(data_total)        
    
    return offs,desv_stand,num_picks,time_peaks_v

def obtienes_disp_gaps_over(st,lista_times_gaps,time_fs,fecha):
    
    num_gaps=0
    num_overlaps=0
    duracion_gap=0

    #---------gaps y over no vistos por obspy (obspy analiza en contenido de la traza, no reporta faltantes de time que no esten fuera de la traza)
    if len(time_fs)>1:
        time_f_dia_anterior=time_fs[-2] #time_fs almacena tiempo final de cada traza. Tiempo final de la del dia anterior es el elemnto [-2]
    else:
        time_f_dia_anterior=fecha #Si no se tiene info del dia anterior, toma la fecha actual a las 00:00, equivale a la hora final del dia anterior 
   
    time_i_st=st[0].stats.starttime.datetime
    time_f_st=st[-1].stats.endtime.datetime
    #print time_i_st,time_f_st,time_f_dia_anterior
    
    #--------mide gaps antes del time_i de la traza
    #---el fin de la traza del dia anterior corresponde al dia ACTUAL
    if fecha.day == time_i_st.day == time_f_dia_anterior.day:    
        del_t_dia_ant_act=(time_i_st - time_f_dia_anterior).total_seconds() #establece gap o overlap con dif entre time_i y time_f del dia anterior
        if del_t_dia_ant_act > 0.1:#lo considera si la diferencia en tiempo es una decima de segundo
            num_gaps=num_gaps+1
            duracion_gap=duracion_gap+del_t_dia_ant_act
            time_i_gap=time_f_dia_anterior
            lista_times_gaps.append([time_i_gap,del_t_dia_ant_act])
        
        if del_t_dia_ant_act < -0.1:
            num_overlaps=num_overlaps+1
    
    #---el fin de la traza del dia anterior corresponde al dia ANTERIOR o antes
    if fecha.day == time_i_st.day != time_f_dia_anterior.day:
        del_t_inicio_dia=(time_i_st - fecha).total_seconds() #establece el gap con time_i respecto a las 00:00 del dia actual
        if del_t_inicio_dia > 0.1:#lo considera si la diferencia en tiempo es una decima de segundo
            num_gaps=num_gaps+1
            duracion_gap=duracion_gap+del_t_inicio_dia
            time_i_gap=fecha
            lista_times_gaps.append([time_i_gap,del_t_inicio_dia])
        

    #-----mide gaps despues del time_f de la traza
    #---el fin de la traza del dia actual corresponde al dia actual
    if fecha.day == time_f_st.day:
        del_t_faltante=( ( fecha + timedelta(days=1) )-time_f_st ).total_seconds() #gap: 00:00 del siguiente dia - time_f del dia actual
        if del_t_faltante > 10:#lo considera si el cierre de la traza del dia actual le falta 10 o mas segundo para alcanzar las 00:00 del otro dia
            num_gaps=num_gaps+1
            duracion_gap=duracion_gap+del_t_faltante
            time_i_gap=time_f_st
            lista_times_gaps.append([time_i_gap,del_t_faltante])
        
    
    
    #---------------------------gaps dentro de traza grabada
    gaps_overlaps=st.get_gaps ()
    if len(gaps_overlaps)>0:
            
        for go in gaps_overlaps:

            if go[-2] > 0: #mayor a cero gap
                num_gaps=num_gaps+1
                duracion_gap=duracion_gap+go[-2]
                time_i_gap=go[-4].datetime

                lista_times_gaps.append([time_i_gap,go[-2]])

            if go[-2] < 0: #menor a cero overlap
                num_overlaps=num_overlaps+1
        
    
    if duracion_gap>0:                     
        disponibilidad_dia_porcentaje=round((1-(duracion_gap/60/60/24))*100,1)
    
    if duracion_gap==0:                
        disponibilidad_dia_porcentaje=100

    return num_gaps,num_overlaps,disponibilidad_dia_porcentaje,lista_times_gaps

if len(ch)==2 and len(estacion_name)>0 and len(servidor_elegido)>4 and time_f>time_i:   
        
    #--- gestiona archivos de salida
    dir_results=os.getcwd()+"/"
    nombre_archivo=dir_results+"Info_Calidad_"+estacion_name+ch+"_"+time_i.strftime("%Y-%m-%d")+"_"+time_f.strftime("%Y-%m-%d")+".txt"
    nombre_archivo_csv=dir_results+"Info_Calidad_"+estacion_name+ch+"_"+time_i.strftime("%Y-%m-%d")+"_"+time_f.strftime("%Y-%m-%d")+".csv"
    nombre_log=dir_results+"Log_"+estacion_name+ch+"_"+time_i.strftime("%Y-%m-%d")+"_"+time_f.strftime("%Y-%m-%d")+".csv"   

    
    archivo_csv=open(nombre_archivo_csv,'w') #    
    reporte=csv.writer(archivo_csv)
    archivo_log=open(nombre_log,'w') #
    archivo_log.writelines("Estacion_canal,Fecha (UTC),Disponibilidad (%),Gaps (conteo),Overlaps (conteo),Offset (cuentas),Picos (conteo)\n")
          
    arc_salida=open(nombre_archivo,"w")
    arc_salida.writelines(estacion_name+ch+"_"+time_i.strftime("%Y-%m-%d")+"_"+time_f.strftime("%Y-%m-%d")+"\n")
  
    data_dic={"Canal": ["Z","N","E"], "Disponibilidad (% dia)": ["","",""], "Numero Gaps (dia)": ["","",""], "Numero Overlaps (dia)": ["","",""],"Offset (Cuentas dia)": ["","",""], "Numero Picos (dia)":["","",""]}
        
    
    #------Hace analisis de Disponibilidad, Gaps, Overlaps, Offset, picos y en el servidor elegido

    print("\nPara el analisis se elige el servidor "+servidor_elegido)    
    ans=range(time_i.year,time_f.year+1)    
    comp=["Z","N","E"]
    #comp=["Z"]

    if estacion_name == "CLBC":#solo tiene la componenete Z
        comp=["Z"]

    if estacion_name == "ROSC": #el canal es BH
        ch = "BH"
    
    i_comp=0        
    for c in comp:
        
        lista_no_grabados=[]
        lista_tiempo=[]
        lista_disponibilidad=[]
        lista_conteo_gaps=[]
        lista_conteo_overlaps=[]
        lista_offset=[]
        lista_desviaciones=[]
        lista_times_gaps=[]
        lista_times_picos=[]
        lista_conteo_picos=[]

        time_fs=[]            
        confirma_canal=0
        for an in ans:            
                
            canal=ch+c
            arc_salida.writelines("\n\n"+canal+"\n")
            direc ="/mnt/"+servidor_elegido+"/"+str(an)+"/CM/"+estacion_name+"/"+canal+".D/"
            
            if os.path.exists(direc):                
                
                confirma_canal=1
                os.chdir(direc)                
                num_dias = (datetime(an, 12, 31) - datetime(an, 1, 1)).days + 1
                vec_dias=np.arange(1,num_dias+1)    
                print ("\n"+str(an) + " " + estacion_name + " " + canal )   
                print ("Evaluando Disponibilidad, gaps, offset y picos en "+direc+"...\n")

                lista_dias=[]
    
                for f in glob.glob("CM.*"):
                    #print f        
                    vecf=f.split(".")
                    dia_jul=int(vecf[-1])
                    lista_dias.append(dia_jul)
                
                arc_basico=f[0:-3]
    
                for v in vec_dias:
                
                    fecha=(datetime(an, 1, 1)+timedelta(days=int(v-1)))
                
                    if fecha > time_f:
                        break
                    if fecha >= time_i:
                        lista_tiempo.append(fecha+timedelta(hours=12))#Se suma medio dia, para mejor aspecto en graficas
            
                        if v in lista_dias:
                            if v<10:            
                                arcFO=arc_basico+"00"+str(v)
                            if 10<=v<100:            
                                arcFO=arc_basico+"0"+str(v)
                            if v>=100:            
                                arcFO=arc_basico+str(v)
                                
                            try:
                                #print "dia",v,fecha
                                st = read(direc+arcFO, format="MSEED")
                                time_f_st=st[-1].stats.endtime.datetime
                                time_fs.append(time_f_st)
                                #print st
                                offs,desv_stand,peaks,time_peaks= obtiene_off_picks(st)                                
                                num_gaps,num_overlaps,disponibilidad_dia_porcentaje,lista_times_gaps=obtienes_disp_gaps_over(st,lista_times_gaps,time_fs,fecha)
                
                                lista_offset.append(offs)
                                lista_desviaciones.append(desv_stand)
                                lista_conteo_picos.append(peaks)                
                                lista_conteo_gaps.append(num_gaps)
                                lista_conteo_overlaps.append(num_overlaps)            
                                lista_disponibilidad.append(disponibilidad_dia_porcentaje)
                
                                for tp in time_peaks:
                                   lista_times_picos.append(tp)

                                archivo_log.writelines(estacion_name+"_"+ch+c+","+fecha.strftime("%Y-%m-%d")+","+str(round(disponibilidad_dia_porcentaje,1))+","+str(num_gaps)+","+str(num_overlaps)+","+str(round(offs,1))+","+str(peaks)+"\n")
                            
                            except Exception as e:
                                print ("No se pudo leer la forma de onda: "+direc+arcFO)
                                print (e)
                                lista_disponibilidad.append(-9)    
                                lista_no_grabados.append([v,fecha])
                                lista_conteo_gaps.append(-9)
                                lista_conteo_overlaps.append(-9)
                                lista_conteo_picos.append(-9)    
                                lista_offset.append(0)
                                lista_desviaciones.append(0)
                                archivo_log.writelines(estacion_name+"_"+ch+c+","+fecha.strftime("%Y-%m-%d")+","+str(-9)+","+str(-9)+","+str(-9)+","+str(0)+","+str(-9)+"\n")
                            
                        if v not in lista_dias:
                            lista_disponibilidad.append(0)    
                            lista_no_grabados.append([v,fecha])
                            lista_conteo_gaps.append(-9)
                            lista_conteo_overlaps.append(-9)
                            lista_conteo_picos.append(-9)    
                            lista_offset.append(0)
                            lista_desviaciones.append(0)
                            lista_times_gaps.append([ datetime(fecha.year,fecha.month,fecha.day,0,0,0),3600*24 ])
                            archivo_log.writelines(estacion_name+"_"+ch+c+","+fecha.strftime("%Y-%m-%d")+","+str(0)+","+str(-9)+","+str(-9)+","+str(0)+","+str(-9)+"\n")
    
            
            else:
                print ("\nNo existe el directorio "+direc)
                print ("Comprobar nombre de la estacion o el canal y su disponibilidad en el año "+str(an))


        if len(lista_no_grabados)>0:
            print ("\nResumen de Dias faltantes "+estacion_name+" "+canal+" entre "+time_i.strftime("%Y-%m-%d")+" y "+time_f.strftime("%Y-%m-%d")+"\n")
            for i in range(0,len(lista_no_grabados)-1):
                print (lista_no_grabados[i][0],lista_no_grabados[i][1].strftime("%Y-%m-%d"))
                if lista_no_grabados[i+1][0]-lista_no_grabados[i][0]>1:
                    print("------- -------")                
            print (lista_no_grabados[-1][0],lista_no_grabados[-1][1].strftime("%Y-%m-%d") )   
        
        
        
        if confirma_canal==1:
            del_time_m=(lista_tiempo[-1]-lista_tiempo[0]).total_seconds()/60/60/24/30  #en meses
            dias_ad=int ( round ( (4*del_time_m)-3 ,0 ) ) #al hacerce mas larga la grafica, debe adicionar mas dias para construir columna de calor (para que se vea)
            if dias_ad <= 0:
                dias_ad=1
          

            #-----GRAFICA DISPONIBILIDAD
            lista_disponibilidad_sin_nueves=[] #es posible que no haya podido leer un mseed por tanto escribio un -9
            for di in lista_disponibilidad:
                if di != -9:
                    lista_disponibilidad_sin_nueves.append(di)
                    
            min_dis=round( min(lista_disponibilidad_sin_nueves),1 )
            max_dis=round( max(lista_disponibilidad_sin_nueves),1 )
            prom_dis=round( np.mean( np.array(lista_disponibilidad_sin_nueves) ),1 )
            arc_salida.writelines("Disponibilidad (% dia): min = "+str(min_dis)+", max = "+str(max_dis)+", prom = "+str(prom_dis) )        
            data_dic["Disponibilidad (% dia)"][i_comp]="min = "+str(min_dis)+"/max = "+str(max_dis)+"/prom = "+str(prom_dis)
          
            fig = plt.figure(figsize=(21., 11.))
            ax = fig.add_subplot(311)
            plt.plot(lista_tiempo,lista_disponibilidad,linewidth=1,color='#0000ff',label="Disponibilidad",zorder=1000)
    
            ax.set_ylabel('Disponibilidad \n (%)', color='k', fontsize=14)
            ax.xaxis.grid(True, which='major')
            ax.yaxis.grid(True, which='major')
            ax.xaxis.set_tick_params(labelsize=9)
            ax.set_xlim( lista_tiempo[0]-timedelta(days=1), lista_tiempo[-1]+timedelta(days=dias_ad))
        
            if len(lista_times_gaps) > 0:
                ax1 = ax.twinx()
                ax1.vlines(lista_times_gaps[0][0],0,0, lw=1, color="k",label="Gaps (tiempo)" )        
                to_r=datetime(1,1,1,0,0,1)
                rectangulo3 = patches.Rectangle( ( lista_tiempo[-1], to_r ),timedelta(dias_ad),timedelta(hours=24),fill=True,facecolor='b',edgecolor='b',linewidth=0.2,alpha=0.6)
                ax1.add_patch(rectangulo3)
                
                for tg in lista_times_gaps:
                    
                    tdia=datetime(tg[0].year,tg[0].month,tg[0].day,12,0,0)
                    to=datetime(1,1,1,tg[0].hour,tg[0].minute,tg[0].second)
                    ax1.text(tdia,to,"o",fontsize=7,alpha=0.8,zorder=100,horizontalalignment='center',verticalalignment='baseline')
                    ax1.text(tdia,to+timedelta(seconds=tg[1]),"x",fontsize=7,alpha=0.8,zorder=100,horizontalalignment='center',verticalalignment='baseline')
                    rectangulo = patches.Rectangle( ( tdia-timedelta(seconds=int(tg[1]/2)), to),timedelta(seconds=tg[1]),timedelta(seconds=tg[1]),fill=True,facecolor='k',edgecolor='k',linewidth=0.6,alpha=0.6, ) 
                    ax1.add_patch(rectangulo)
            
                    if tg[1] < 3600*24:#si el gap fue de todo el dia no lo grafica en columna calor
                        rectangulo2 = patches.Rectangle( ( lista_tiempo[-1], to),timedelta(dias_ad),timedelta(seconds=tg[1]),fill=True,facecolor='k',edgecolor='k',linewidth=0.2,alpha=0.08, ) 
                        ax1.add_patch(rectangulo2)
        
            
                ax1.set_ylim( datetime(1,1,1,0,0,0), datetime(1,1,1,23,59,59))    
                ax1.yaxis.set_minor_locator( HourLocator(interval = int(2)))
                ax1.yaxis.set_minor_formatter( DateFormatter('%H:%m') )
                ax1.yaxis.set_major_locator( HourLocator(interval = int(8)))
                ax1.yaxis.set_major_formatter( DateFormatter('%H') )       

                labels = ax1.yaxis.get_minorticklabels()
                plt.setp(labels, rotation=0, fontsize=8)
                labels = ax1.get_yticklabels() 
                plt.setp(labels, rotation=0, fontsize=0)    
        
                ax1.set_ylabel('Gaps\nHora (UTC)', color='k', fontsize=14)
                ax1.legend(loc='upper right')
            
            ax.legend(loc='upper left')        
            plt.title('Disponibilidad ' +estacion_name+ " "+ canal+ " "+time_i.strftime("%Y-%m-%d")+" - "+time_f.strftime("%Y-%m-%d"))
    
        
            #-----GRAFICA CONTEO DE GAPS, OVERLAPS y offset
            #-----GRAFICA OFFSET
            ax = fig.add_subplot(312)
            #cuando una estacion entra genera un offset muy grande ese dia, si por ejemplo viene un offet promedio de 2000, pero el dia que entra             nuevamente la estacion ese offset es de 200000, entonces se ve plana la grafica de offset y un pico en el 200000,Para mejorar la             visualizacion, se pone un offset cero el dia que se reconoce que entra la estacion.
        
            busca_ceros=[]
            i=0
            for e in lista_offset:
                if e==0:
                    busca_ceros.append(i)    
                if len(busca_ceros)>0:
                    if e!=0:
                        lista_offset[i]=0 #al siguiente dia del cero, tambien pone un cero,
                        busca_ceros=[]    
                i=i+1
            #------------------------
           
            lista_offset_sin_ceros=[]
            for of in lista_offset:
                if of != 0:
                    lista_offset_sin_ceros.append(of)
            offset_prom=round(np.mean(np.array(lista_offset_sin_ceros)),2)

            arc_salida.writelines("\nOffset (cuentas): max (dia) = "+str( round (max(lista_offset_sin_ceros),2) )+", min (dia) = "+str( round (min(lista_offset_sin_ceros),2) )+", promedio (dia) = "+str(round(offset_prom,2) ) )
            data_dic["Offset (Cuentas dia)"][i_comp]="max = "+str( round (max(lista_offset_sin_ceros),2) )+"/min = "+str( round (min(lista_offset_sin_ceros),2) )+"/prom = "+str(round(offset_prom,2) )
            
            #para mejorar visalizacion offset. Los ceros puestos que son cuando no se pudo calcular (por no disponibilidad)
            #los convierte en el promedio. Si existen valores gigantes, 50 veces mayor a la media grafica el offset en escala log
        
            escala="linear"
            i=0
            for e in lista_offset:
                if e==0 :
                    lista_offset[i]=offset_prom
                if abs(e)>abs(50*offset_prom) or abs(e)>100000:
                    escala="log"
                i=i+1
            #-----------

            if escala == "linear":
                ax.plot(lista_tiempo,lista_offset,linewidth=1,color="#ff0000",label="Offset")        
                ax.hlines(offset_prom, min(lista_tiempo), max(lista_tiempo), colors='k', linestyles='--', linewidth=0.5)
                ax.text(lista_tiempo[int(len(lista_tiempo)/2)], offset_prom, "media: "+str(offset_prom), verticalalignment='top',fontsize=9)   
                ax.set_ylabel('Offset', color='k', fontsize=14)
            if escala == "log": #si esta en escala lineal obtiene el abs del offset (para poder graficar valores negativos del offset original, no                     grafica media porque pierde sentido, esto sucede porque hay valores gigantes respecto a la media)
                ax.plot(lista_tiempo,np.abs(np.array(lista_offset)),linewidth=1,color="#ff0000",label="Offset")         
                ax.set_ylabel('abs (Offset)', color='k', fontsize=14)

            ax.legend(loc='upper left')        
            ax.xaxis.grid(True, which='major')
            ax.yaxis.grid(True, which='major')
            ax.xaxis.set_tick_params(labelsize=9)
            ax.set_yscale(escala)
            ax.set_xlim( lista_tiempo[0]-timedelta(days=1), lista_tiempo[-1]+timedelta(days=dias_ad))
        
            #overlaps
            ax1 = ax.twinx()
        
            lista_overlaps_sin_nueves=[]
            for i in range(0,len(lista_tiempo)):
                if lista_conteo_overlaps[i] != -9: #se le puso -9 a los datos no disponibles
                    lista_overlaps_sin_nueves.append(lista_conteo_overlaps[i])                
                    ax1.vlines(lista_tiempo[i]-timedelta(minutes=5), 0, lista_conteo_overlaps[i], colors='#32cd32', linewidth=1.5,alpha=0.8)
            
            ax1.set_ylabel('Conteo de \n(Overlaps, Gaps)', color='k', fontsize=14)
            ax1.xaxis.grid(True, which='major')
            ax1.yaxis.grid(True, which='major')
            ax1.xaxis.set_tick_params(labelsize=9)
            ax1.vlines(lista_tiempo[0], 0, 0, colors='#32cd32', linewidth=1.5, label='Overlaps')        
        
            overlaps_prom=np.mean(np.array(lista_overlaps_sin_nueves))
            
            arc_salida.writelines("\nNumero Overlaps: max (dia) = "+str( max(lista_overlaps_sin_nueves) )+", promedio (dia) = "+str(round(overlaps_prom,2) ) )
            data_dic["Numero Overlaps (dia)"][i_comp]="max = "+str( max(lista_overlaps_sin_nueves) )+"/prom = "+str(round(overlaps_prom,2))

            
            # gaps
        
            lista_gaps_sin_nueves=[]
            for i in range(0,len(lista_tiempo)):
                if lista_conteo_gaps[i] != -9: #se le puso -9 a los datos no disponibles
                    lista_gaps_sin_nueves.append(lista_conteo_gaps[i])                
                    ax1.vlines(lista_tiempo[i]+timedelta(minutes=5), 0, lista_conteo_gaps[i], colors='k', linewidth=1.5,alpha=0.8)
            
            ax1.vlines(lista_tiempo[-1], 0, 0, colors='k', linewidth=1.5, label='Gaps')            
            ax1.legend(loc='upper right')
            plt.title('\n\n\nConteo de (Gaps - Overlaps) y Offset ' + estacion_name + " " + canal +" "+time_i.strftime("%Y-%m-%d")+" - "+time_f.strftime("%Y-%m-%d"))
            
            gaps_prom=np.mean(np.array(lista_gaps_sin_nueves))
            arc_salida.writelines("\nNumero Gaps: max (dia) = "+str( max(lista_gaps_sin_nueves) )+", promedio (dia) = "+str(round(gaps_prom,2) ) )
            data_dic["Numero Gaps (dia)"][i_comp]="max = "+str( max(lista_gaps_sin_nueves) )+"/prom = "+str(round(gaps_prom,2))
            
            
            #-------GRFICA DE PICOS (CONTEO Y TIEMPO)

            ax = fig.add_subplot(313)
            lista_picos_sin_nueves=[]
            for i in range(0,len(lista_tiempo)):
                if lista_conteo_picos[i] != -9: #se le puso -9 a los datos no disponibles
                    lista_picos_sin_nueves.append(lista_conteo_picos[i])                
                    ax.vlines(lista_tiempo[i], 0, lista_conteo_picos[i], colors='#8000ff', linewidth=1.0,alpha=1,zorder=100)
            
            ax.vlines(lista_tiempo[-1], 0, 0, colors='#8000ff', linewidth=1.0, label='Picos (conteo)')        
            ax.set_ylabel('Conteo de \nPicos', color='k', fontsize=14)                
            ax.legend(loc='upper left')                   
            plt.title('\n\n\n Picos ' + estacion_name + " " + canal + " "+time_i.strftime("%Y-%m-%d")+" - "+time_f.strftime("%Y-%m-%d")) 
            ax.xaxis.grid(True, which='major')        
            ax.yaxis.grid(True, which='major')   
            ax.xaxis.set_tick_params(labelsize=9)
            ax.set_xlim( lista_tiempo[0]-timedelta(days=1), lista_tiempo[-1]+timedelta(days=dias_ad))
            picos_prom=np.mean(np.array(lista_picos_sin_nueves))
            arc_salida.writelines("\nNumero Picos: max (dia) = "+str( max(lista_picos_sin_nueves) )+", promedio (dia) = "+str(round(picos_prom,2) ) )
            data_dic["Numero Picos (dia)"][i_comp]="max = "+str( max(lista_picos_sin_nueves) )+"/prom = "+str(round(picos_prom,2) )
        
            #EN TIEMPO
            
            if len(lista_times_picos) > 0:
                ax1 = ax.twinx()

                to_r=datetime(1,1,1,0,0,1)
                rectangulo3 = patches.Rectangle( ( lista_tiempo[-1], to_r ),timedelta(dias_ad),timedelta(hours=24),fill=True,facecolor='b',edgecolor='b',linewidth=0.2,alpha=0.8)
                ax1.add_patch(rectangulo3)        

                for tp in lista_times_picos:
                    tdia=datetime(tp.year,tp.month,tp.day,12,0,0)
                    to=datetime(1,1,1,tp.hour,tp.minute,tp.second)                    
                    ax1.scatter(tdia,to,s=1.5,c="k",alpha=0.5,zorder=100)
                    ax1.hlines(to, tdia, max(lista_tiempo), colors='k', linestyles='--', linewidth=0.1,alpha=0.7) 
                    ax1.hlines(to, lista_tiempo[-1], lista_tiempo[-1]+timedelta(dias_ad), colors='k', linestyles='-', linewidth=0.2,alpha=0.7)

                ax1.scatter(tdia,to,s=1,c="k",alpha=0.5,label="Picos (tiempo)")                        
                ax1.set_ylim( datetime(1,1,1,0,0,0), datetime(1,1,1,23,59,59))    
                ax1.yaxis.set_minor_locator( HourLocator(interval = int(2)))
                ax1.yaxis.set_minor_formatter( DateFormatter('%H:%m') )
                ax1.yaxis.set_major_locator( HourLocator(interval = int(8)))
                ax1.yaxis.set_major_formatter( DateFormatter('%H') )          
                    
                labels = ax1.yaxis.get_minorticklabels()
                plt.setp(labels, rotation=0, fontsize=8)
                labels = ax1.get_yticklabels() 
                plt.setp(labels, rotation=0, fontsize=0)    
        
                ax1.set_ylabel('Picos\nHora (UTC)', color='k', fontsize=14)
                ax1.legend(loc='upper right')
            
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(dir_results+"Cal_"+estacion_name+"_"+canal+".png", bbox_inches='tight')
            i_comp=i_comp+1

    
    arc_salida.close()
              
    #------
    #escribe el csv
    for key in data_dic:
        lista=[]
        lista.append(key)
        values=data_dic[key]
        for v in values:
            lista.append(v)
        reporte.writerow(lista)
        
    archivo_csv.close()
    archivo_log.close()
    
    #-----    
    print ("\n Graficando \n")

    plt.show()

else:
    print ("No se pudo ejecutar el programa. Revise el archivo de entrada")




    
    
    
    
        
