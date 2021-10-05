import sys
import epics
import epics.devices
import numpy as np
import time
import h5py
import netCDF4
import os
import fabio
from ljm_functions import *
from multiprocessing import Process
from readMDA import *

scanrecord = "26idbSOFT"

ljm_scaler = T7_ID26(3)
ljm_fluo = T7_ID26(2) 
ljm_mpx = T7_ID26(1) 
ljm_eiger = T7_ID26(4)


def get_detector_flags():

    flag_fluo = 0
    flag_eiger = 0
    for i in range(1,5):
        det_name = epics.caget("26idbSOFT:scan1.T{0}PV".format(i))
        if "XMAP" in det_name or "scanH" in det_name:
            flag_fluo = 1
        elif 's26_eiger' in det_name:
            flag_eiger = 1
    return flag_fluo, flag_eiger


def flyxy_head(n_line, pts_per_line):

    flag_fluo, flag_eiger = get_detector_flags()

    epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 1, wait=True) # set fly flag = 1

    epics.caput("26idpvc:userCalc3.SCAN", 0, wait=True) # disable Martin's samy watchdog

    t0 = time.time()

    pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)[:-4]
    scannum = epics.caget(scanrecord+':saveData_scanNumber')
    epics.caput(scanrecord+':saveData_scanNumber', scannum+1)

    if not flag_eiger: # have to create the h5 file if eiger is not doing it
        pathname_h5 = os.path.join(pathname,"h5")
        filenum =  epics.caget("s26_eiger_cnm:HDF1:FileNumber_RBV")
        h5file = os.path.join(pathname, "h5", "scan_{0}_{1:06d}.h5".format(scannum,filenum))
        h5 = h5py.File(h5file, "w-")
        h5.close()

    print ("starting flyscan {0}\nactivating detectors....".format(scannum)),

    if flag_eiger:

        if epics.caget("s26_eiger_cnm:HDF1:FilePath", as_string=True) != (pathname+'h5/'):
            epics.caput("s26_eiger_cnm:HDF1:FilePath",os.path.join(pathname,'h5/'), wait=True)
        if epics.caget("s26_eiger_cnm:HDF1:EnableCallbacks") == 0:
            epics.caput("s26_eiger_cnm:HDF1:EnableCallbacks",1, wait=True)
        if epics.caget("s26_eiger_cnm:cam1:TriggerMode") != 3:
            epics.caput("s26_eiger_cnm:cam1:TriggerMode",3, wait=True)
        epics.caput("s26_eiger_cnm:HDF1:FileName",'scan_'+str(scannum), wait=True)
        epics.caput("s26_eiger_cnm:HDF1:NumCapture", n_line*pts_per_line, wait=True)
        epics.caput("s26_eiger_cnm:cam1:NumTriggers", n_line*pts_per_line, wait=True)
        epics.caput("s26_eiger_cnm:HDF1:Capture", 1)
        epics.caput("s26_eiger_cnm:cam1:Acquire", 1)
        time.sleep(0.1)
        
        epics.caput("s26_eiger_cnm:Stats1:TS:TSNumPoints", n_line*pts_per_line, wait=True)
        epics.caput("s26_eiger_cnm:Stats2:TS:TSNumPoints", n_line*pts_per_line, wait=True)
        epics.caput("s26_eiger_cnm:Stats3:TS:TSNumPoints", n_line*pts_per_line, wait=True)
        epics.caput("s26_eiger_cnm:Stats4:TS:TSNumPoints", n_line*pts_per_line, wait=True)
        epics.caput("s26_eiger_cnm:Stats1:TS:TSAcquire", 1)
        time.sleep(0.1)
        epics.caput("s26_eiger_cnm:Stats2:TS:TSAcquire", 1)
        time.sleep(0.1)
        epics.caput("s26_eiger_cnm:Stats3:TS:TSAcquire", 1)
        time.sleep(0.1)
        epics.caput("s26_eiger_cnm:Stats4:TS:TSAcquire", 1)
        time.sleep(0.1)
        
    else:
        epics.caput('s26_eiger_cnm:HDF1:FileNumber', filenum+1, wait=True) # increment filenum +1 if eiger not saved

    #epics.caput("26idc:3820:scaler1.CONT", 1) # enable scaler auto count for normalization
    #epics.caput("26idc:3820:scaler1.TP", 1)
    epics.caput("26idc:3820:ChannelAdvance", 1, wait=True) # set scaler to external trigger
    epics.caput("26idc:3820:Channel1Source", 1, wait=True) 
    epics.caput("26idc:3820:InputMode", 3, wait=True) # set scaler input mode to 3
    epics.caput("26idc:3820:NuseAll", pts_per_line, wait=True) # set channel to use to the nbpoints per line
    epics.caput("26idc:3820:PresetReal", 10000, wait=True) # set it to very long
    epics.caput("26idc:3820:scaler1.CONT", 0) # disable auto count, or it will get stuck

    if flag_fluo:
        epics.caput("26idcXMAP:CollectMode", 1, wait=True) #mca mapping
        epics.caput("26idcXMAP:PixelsPerRun", pts_per_line, wait=True) 
        epics.caput("26idcXMAP:IgnoreGate", 0, wait=True) #very important
        pathname_fluo = os.path.join(pathname, "fluo")
        if not os.path.isdir(pathname_fluo):
            os.mkdir(pathname_fluo)
            os.chmod(pathname_fluo, 0o777)
        pathname_fluo = "T:"+pathname_fluo[5:]
        epics.caput("26idcXMAP:netCDF1:EnableCallbacks", 1, wait=True) # enable netcdf saving
        epics.caput("26idcXMAP:netCDF1:FilePath", pathname_fluo, wait=True)
        epics.caput("26idcXMAP:netCDF1:FileName", "scan_"+str(scannum), wait=True)
        epics.caput("26idcXMAP:netCDF1:FileTemplate", "%s%s_%6.6d.nc", wait=True)
        epics.caput("26idcXMAP:netCDF1:FileWriteMode", 2, wait=True) # stream
        epics.caput("26idcXMAP:netCDF1:NumCapture", n_line, wait=True)  
        epics.caput("26idcXMAP:netCDF1:Capture", 1, wait=False) # capture fluo
        epics.caput("26idcXMAP:netCDF1:AutoIncrement", 1, wait=False) 
        epics.caput("26idcXMAP:StartAll", 1)
        time.sleep(0.1)
    
    while((epics.caget("s26_eiger_cnm:HDF1:Capture")==0 and flag_eiger) or\
          (epics.caget("26idcXMAP:netCDF1:Capture_RBV")==0 and flag_fluo)):
        time.sleep(1)

    ljm_scaler.write("DAC0", 3.3) # scaler is active low, need to set it to high at the beginning

    epics.caput("26idc:filter:Fi1:Set",0, wait=True)
    time.sleep(0.1)

    print ("completed in {0:.1f} sec".format(time.time()-t0))
    

def flyxy_tail(fastmotor, fastmotorvalues, slowmotor, slowmotorvalues, dwelltime, data_scaler):

    flag_fluo, flag_eiger = get_detector_flags()

    print ("returning config to normal...")

    pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)[:-4]

    epics.caput("26idc:filter:Fi1:Set",1)
    time.sleep(.1)

    ljm_scaler.write("DAC0", 0) 
    epics.caput("26idc:3820:ChannelAdvance", 0, wait=True) # set scaler to internal trigger
    epics.caput("26idc:3820:Channel1Source", 0, wait=True) 
    epics.caput("26idc:3820:PresetReal", 1, wait=True) 

    if flag_eiger:
        if epics.caget("s26_eiger_cnm:HDF1:Capture_RBV"):
            print("It seems that Eiger did not capture all the expected images, did you interrupt?")
            epics.caput("s26_eiger_cnm:HDF1:Capture", 0)
            time.sleep(.1)
        if epics.caget("s26_eiger_cnm:cam1:Acquire") == 1:
            epics.caput("s26_eiger_cnm:cam1:Acquire", 0)
            time.sleep(.1)
        if epics.caget("s26_eiger_cnm:Stats1:TS:TSAcquiring"):
            epics.caput("s26_eiger_cnm:Stats1:TS:TSAcquire", 0)
            time.sleep(0.1)
        if epics.caget("s26_eiger_cnm:Stats2:TS:TSAcquiring"):
            epics.caput("s26_eiger_cnm:Stats2:TS:TSAcquire", 0)
            time.sleep(0.1)
        if epics.caget("s26_eiger_cnm:Stats3:TS:TSAcquiring"):
            epics.caput("s26_eiger_cnm:Stats3:TS:TSAcquire", 0)
            time.sleep(0.1)
        if epics.caget("s26_eiger_cnm:Stats4:TS:TSAcquiring"):
            epics.caput("s26_eiger_cnm:Stats4:TS:TSAcquire", 0)
            time.sleep(0.1)
        
    if flag_fluo:
        epics.caput("26idcXMAP:netCDF1:Capture", 0) # capture fluo
        time.sleep(.1)
        epics.caput("26idcXMAP:StopAll", 1) 
        time.sleep(.1)
        epics.caput("26idcXMAP:netCDF1:EnableCallbacks", 0, wait=True) # disable netcdf saving
        epics.caput("26idcXMAP:IgnoreGate", 1, wait=True) 
        epics.caput("26idcXMAP:CollectMode", 0, wait=True) # has to be put in last or it does not work it seems

    print ("appending metadata in h5..."),
    t0 = time.time()

    pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)[:-4]
    scannum = epics.caget(scanrecord+':saveData_scanNumber')-1
    filenum =  epics.caget("s26_eiger_cnm:HDF1:FileNumber_RBV")-1
    h5file = os.path.join(pathname, "h5", "scan_{0}_{1:06d}.h5".format(scannum,filenum))
    h5 = h5py.File(os.path.join(pathname, "h5", "scan_{0}_{1:06d}.h5".format(scannum,filenum)), "r+")

    if flag_eiger:
        inst_grp = h5["/entry/instrument"]
    else:
        inst_grp = h5.create_group("/entry/instrument")
    sr_grp = inst_grp.create_group("Storage Ring")
    sr_grp.create_dataset("SR current", data=epics.caget('S:SRcurrentAI'))
    ida_grp = inst_grp.create_group("26-ID-A")
    for i in range(1,5):
        ida_grp.create_dataset(epics.caget('26idaWBS:m{0}.DESC'.format(i)), data=epics.caget('26idaWBS:m{0}.RBV'.format(i)))
    for i in range(1,9):
        ida_grp.create_dataset(epics.caget('26idaMIR:m{0}.DESC'.format(i)), data=epics.caget('26idaMIR:m{0}.RBV'.format(i)))
    for i in range(1,5):
        ida_grp.create_dataset(epics.caget('26idaBDA:m{0}.DESC'.format(i)), data=epics.caget('26idaBDA:m{0}.RBV'.format(i)))
    idb_grp = inst_grp.create_group("26-ID-B")
    for i in range(1,9):
        idb_grp.create_dataset(epics.caget('26idbDCM:sm{0}.DESC'.format(i)), data=epics.caget('26idbDCM:sm{0}.RBV'.format(i)))
    for i in range(1,5):
        idb_grp.create_dataset(epics.caget('26idbPBS:m{0}.DESC'.format(i)), data=epics.caget('26idbPBS:m{0}.RBV'.format(i)))
    idc_grp = inst_grp.create_group("26-ID-C")
    idc_grp.create_dataset("count_time", data=dwelltime)
    for i in range(3,6):
        idc_grp.create_dataset(epics.caget('26idcSOFT:sm{0}.DESC'.format(i)), data=epics.caget('26idcSOFT:sm{0}.RBV'.format(i)))
    for i in range(1,13):
        idc_grp.create_dataset(epics.caget('26idcDET:m{0}.DESC'.format(i)), data=epics.caget('26idcDET:m{0}.RBV'.format(i)))
    
    for i in range(1,5):
        idc_grp.create_dataset(epics.caget('atto2:m{0}.DESC'.format(i)), data=epics.caget('atto2:m{0}.RBV'.format(i)))
    for i in range(1,18):
        idc_grp.create_dataset(epics.caget('26idcnpi:m{0}.DESC'.format(i)), data=epics.caget('26idcnpi:m{0}.RBV'.format(i)))
    for i in range(34,36):
        idc_grp.create_dataset(epics.caget('26idcnpi:m{0}.DESC'.format(i)), data=epics.caget('26idcnpi:m{0}.RBV'.format(i)))
    idc_grp.create_dataset(epics.caget('atto2:PIC867:1:m1.DESC'.format(i)), data=epics.caget('atto2:PIC867:1:m1.RBV'.format(i)))
    idc_grp.create_dataset(epics.caget('26idcnpi:Y_HYBRID_SP.DESC'.format(i)), data=epics.caget('26idcnpi:Y_HYBRID_SP.VAL'.format(i)))
    idc_grp.create_dataset(epics.caget('26idcnpi:X_HYBRID_SP.DESC'.format(i)), data=epics.caget('26idcnpi:X_HYBRID_SP.VAL'.format(i)))
    idc_grp.create_dataset('NES H Slit', data=epics.caget('26idcNES:Slit1Hsize.VAL'))
    idc_grp.create_dataset('NES V Slit', data=epics.caget('26idcNES:Slit1Vsize.VAL'))

    ny, nx = fastmotorvalues.shape

    dim = [{}]
    rank = 2
    dim[0]['version'] = 1.3
    dim[0]['scan_number'] = scannum
    dim[0]['rank'] = rank
    dim[0]['dimensions'] = [ny, nx]
    dim[0]['isRegular'] = 1
    dim[0]['time'] = 0
    dim[0]['ourKeys'] = ['ourKeys', 'version', 'scan_number', 'rank', 'dimensions', 'isRegular', 'time']

    for i in range(1,rank+1):
        dim.append(scanDim())
        dim[i].dim = i
        dim[i].rank = rank-i+1
        dim[i].scan_name = '26idbSOFT:scan{0}'.format(i)
        dim[i].np = 1
        dim[i].time = "whatever"
    dim[1].nd = 0
    dim[1].npts = ny
    dim[1].curr_pt = ny
    dim[1].p.append(scanPositioner())
    if slowmotor == "hybridy":
        dim[1].p[0].name = "26idcnpi:Y_HYBRID_SP.VAL"
        dim[1].p[0].desc = 'Hybrid Piezo Y'
    elif slowmotor == "attoz":
        dim[1].p[0].name = "26idbATTO:m3.VAL"
        dim[1].p[0].desc = 'ATTO SAM Z'
    else:
        dim[1].p[0].name = "26idbATTO:m4.VAL"
        dim[1].p[0].desc = 'ATTO SAM X'
    dim[1].p[0].data = slowmotorvalues

    dim[2].p.append(scanPositioner())
    dim[2].npts = nx
    dim[2].nd = 70
    dim[2].curr_pt = nx
    if fastmotor == "hybridx":
        dim[2].p[0].name = "26idcnpi:X_HYBRID_SP.VAL"
        dim[2].p[0].desc = 'Hybrid Piezo X'
    else:
        dim[2].p[0].name = "26idcnpi:m17.VAL"
        dim[2].p[0].desc = 'Sample Y'
    dim[2].p[0].data = fastmotorvalues
    
    print("completed in {0:0.1f} sec".format(time.time()-t0))
    t0 = time.time()

    if flag_fluo:
        print("extracting fluo spectra from netcdf..."),
        fluonum =  epics.caget("26idcXMAP:netCDF1:FileNumber_RBV")-1
        f_fluo = os.path.join(pathname, "fluo",  "scan_{0}_{1:06d}.nc".format(scannum,fluonum))
        os.chmod(f_fluo, 0o777)
        netcdffile = netCDF4.Dataset(f_fluo, "r")
        data_mca_ib = (netcdffile.variables['array_data'][:,0,256:].reshape(ny, 124, 256+2048*4)[:,:nx,256:].reshape(ny,nx,4,2048))
        data_mca_ob = (netcdffile.variables['array_data'][:,1,256:].reshape(ny, 124, 256+2048*4)[:,:nx,256:].reshape(ny,nx,4,2048))[:,:,-1]
        netcdffile.close()
        print("completed in {0:0.1f} sec".format(time.time()-t0))
        t0 = time.time()

    print("calculating roi stats for h5..."),

    pos_grp = h5.create_group("/entry/scan/Positioner")
    h5["/entry/scan"].attrs["dimensions"] = (ny,nx)
    dset = pos_grp.create_dataset(slowmotor, data=slowmotorvalues)
    dset = pos_grp.create_dataset(fastmotor, data=fastmotorvalues)
    dset = pos_grp.create_dataset("dwelltime", data=dwelltime)

    det_grp = h5.create_group("/entry/scan/Detector")
    for i in range(70):
        name = epics.caget("26idbSOFT:scan1.D{0:02d}PV".format(i+1))
        dset = det_grp.create_dataset("D{0:02d}:{1}".format(i+1, name), data=np.zeros(fastmotorvalues.shape), dtype="f8")
        dim[2].d.append(scanDetector())
        dim[2].d[i].name = name
        dim[2].d[i].number = i
        if "s26_eiger_cnm:Stats" in name:
            i_roi = name[19]
            xmin = epics.caget("s26_eiger_cnm:ROI{0}:MinX".format(i_roi)) 
            xmax = min(epics.caget("s26_eiger_cnm:ROI{0}:SizeX".format(i_roi)) + xmin, 1027)
            ymin = epics.caget("s26_eiger_cnm:ROI{0}:MinY".format(i_roi))
            ymax = min(epics.caget("s26_eiger_cnm:ROI{0}:SizeY".format(i_roi)) + ymin, 1061)
            dset.attrs['xmin'] = xmin
            dset.attrs['xmax'] = xmax
            dset.attrs['ymin'] = ymin
            dset.attrs['ymax'] = ymax
            if "Total" in name:
                dset[...] = epics.caget("s26_eiger_cnm:Stats{0}:TSTotal".format(i_roi))[:ny*nx].reshape(ny,nx)
            elif "CentroidX" in name:
                dset[...] = epics.caget("s26_eiger_cnm:Stats{0}:TSCentroidX".format(i_roi))[:ny*nx].reshape(ny,nx)
            elif "CentroidY" in name:
                dset[...] = epics.caget("s26_eiger_cnm:Stats{0}:TSCentroidY".format(i_roi))[:ny*nx].reshape(ny,nx)
        elif "userCalcOut" in name and flag_fluo:
            i_roi = int(name[21:23])
            xmin = epics.caget("26idcXMAP:mca1.R{0}LO".format(i_roi))
            xmax = epics.caget("26idcXMAP:mca1.R{0}HI".format(i_roi))
            dset.attrs["xmin"] = xmin
            dset.attrs["xmax"] = xmax
            dset.attrs["line"] = epics.caget("26idcXMAP:mca1.R{0}NM".format(i_roi))
            dset[...] = data_mca_ib[:,:,:,xmin:xmax+1].sum(2).sum(2)
        elif "mca8.R" in name and flag_fluo:
            i_roi = name[16:]
            xmin = epics.caget("26idcXMAP:mca8.R{0}LO".format(i_roi))
            xmax = epics.caget("26idcXMAP:mca8.R{0}HI".format(i_roi))
            dset.attrs["xmin"] = xmin
            dset.attrs["xmax"] = xmax
            dset.attrs["line"] = epics.caget("26idcXMAP:mca8.R{0}NM".format(i_roi))
            dset[...] = data_mca_ob[:,:,xmin:xmax+1].sum(2)
        elif "scaler1_cts1.B" in name:
            dset[...] = data_scaler[0]
        elif "scaler1_cts1.C" in name:
            dset[...] = data_scaler[1]
            
        dim[2].d[i].data = dset[()]

    data_mca_ib = None
    data_mca_ob = None
    h5.close()
    print("completed in {0:0.1f} sec".format(time.time()-t0))

    f_mda = os.path.join(pathname, "mda", "26idbSOFT_{0:04d}.mda".format(scannum))

    writeMDA(dim, f_mda)
    print("{0} is now available.".format(f_mda)),

    epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 0, wait=True) # set fly flag = 0
    epics.caput("26idpvc:userCalc3.SCAN", 6, wait=True) # enable Martin's samy watchdog


def flyxy_sample(dx0, dx1, nx, dy0, dy1, ny, dwelltime, delaytime=0.0, velofactor=0.92, slowmotor='attoz'):

    data_scaler = np.zeros((2,nx,ny))

    flag_fluo, flag_eiger = get_detector_flags()

    if ny>124 and flag_fluo:
        sys.exit("number of points must be smaller than 124 for the fast axis...")
    if nx*ny > 16000:
        sys.exit("number of total points must be smaller than 16000...")

    fm_spd = np.abs(dy0-dy1)*1./ny/dwelltime
    if fm_spd < 10 or fm_spd > 50:
        sys.exit("hybridx motor speed must be between 10 and 50 um/s, currently {0:.1f}".format(fm_spd))

    while(epics.caget("PA:26ID:SCS_BLOCKING_BEAM.VAL") or epics.caget("PA:26ID:FES_BLOCKING_BEAM.VAL")):
        print("it seems that either the A or the C hutch shutter is closed, checking again in 1 minute")
        time.sleep(60)

    if epics.caget("26idc:1:userCalc7.VAL") or epics.caget("26idc:1:userCalc5.VAL"):
        sys.exit("please unlock hybrid before performing this kind of scan")

    if slowmotor == 'attoz':
        xorigin = epics.caget("atto2:m3.VAL")
    elif slowmotor == 'attox':
        xorigin = epics.caget("atto2:m4.VAL")
    xstart = dx0 + xorigin
    xend = dx1 + xorigin
    yorigin = epics.caget("26idcnpi:m17.VAL")
    ystart = dy0 + yorigin
    yend = dy1 + yorigin

    epics.caput("26idcnpi:m17.VAL", ystart, wait=True)
    if slowmotor == 'attoz':
        epics.caput("atto2:m3.VAL", xstart, wait=True) # turns out the wait works, no need to add DMOV
    elif slowmotor == 'attox':
        epics.caput("atto2:m4.VAL", xstart, wait=True) # turns out the wait works, no need to add DMOV
    epics.caput("26idcnpi:m17.VELO", np.fabs(dy0-dy1)/ny/dwelltime*velofactor, wait=True)
    abs_x = np.linspace(xstart, xend, nx)
    abs_y = np.linspace(ystart, yend, ny)

    flyxy_head(nx, ny)

    hiccups = []

    print("scanning line"),

    def process1():
        time.sleep(delaytime+0.005) # before this got launched first
        ljm_fluo.send_fakedigital_singlebuffer(np.ones(ny), t1=dwelltime, out_num=0, dac_num=0, t0=0.005)
    def process2():
        time.sleep(delaytime)
        ljm_scaler.send_fakedigital_singlebuffer(np.ones(ny+1), t1=dwelltime, out_num=0, dac_num=0, t0=0.005, inverted=True)
    def process3():
        time.sleep(delaytime)
        ljm_mpx.send_fakedigital_singlebuffer(np.ones(ny), t1=dwelltime, out_num=0, dac_num=0, t0=0.01)
    def process4():
        time.sleep(delaytime)
        ljm_eiger.send_fakedigital_singlebuffer(np.ones(ny), t1=dwelltime, out_num=0, dac_num=0, t0=0.005) 

    for i_x in range(nx):

        epics.caput("26idc:3820:EraseStart",1)
        time.sleep(0.1)

        hiccup = 0
        print("{0:03d}/{1:03d}".format(i_x+1,nx)+"\b"*8),

        p1 = Process(target=process1)
        p4 = Process(target=process4)
        p2 = Process(target=process2)        
   
        if flag_fluo:
            p1.start()
        if flag_eiger:
            p4.start()
        p2.start()

        epics.caput("26idcnpi:m17.VAL", yend, wait=True)

        #time.sleep(nx * dwelltime+.2)
        flag_break = False

        while(1):
            if flag_fluo:
                if epics.caget("26idcXMAP:Acquiring"):
                    hiccup += 1
                    print("waiting on fluo")
                    time.sleep(dwelltime)
                    flag_break = True
            if flag_eiger:
                num_cap = epics.caget("s26_eiger_cnm:HDF1:NumCaptured_RBV")
                if num_cap != (i_x+1)*ny:
                    hiccup += 1
                    print("waiting on eiger ", num_cap)
                    time.sleep(dwelltime)
                    flag_break = True
            if epics.caget("26idc:3820:Acquiring"):
                hiccup += 1
                print("waiting on scaler")
                time.sleep(dwelltime)
                flag_break = True
                if hiccup > 20:
                    print("stopping scaler")
                    epics.caput("26idc:3820:StopAll",1)
            if not flag_break:
                break
            flag_break = False

 
        if slowmotor == 'attoz':
            epics.caput("atto2:m3.VAL", abs_x[i_x])
        elif slowmotor == 'attox':
            epics.caput("atto2:m4.VAL", abs_x[i_x])
        epics.caput("26idcnpi:m17.VELO", 200, wait=True)
        epics.caput("26idcnpi:m17.VAL", ystart, wait=True)
        epics.caput("26idcnpi:m17.VELO", np.fabs(dy0-dy1)/ny/dwelltime*velofactor, wait=True)
        #time.sleep(0.5) # to make sure that the data is saved, putting 0 will cause some data to not be updated!!!!

        if hiccup:
            hiccups+=[[i_x, hiccup]]

        for i_scaler in range(2):
            data_scaler[i_scaler, i_x] = epics.caget('26idc:3820:mca{0}'.format(i_scaler+2))[:ny]

        if flag_fluo:
            epics.caput("26idcXMAP:StartAll", 1)

    epics.caput("26idcnpi:m17.VELO", 200, wait=True)

    ljm_fluo.stop_streaming()
    ljm_scaler.stop_streaming()
    #ljm_mpx.stop_streaming()
    ljm_eiger.stop_streaming()

    epics.caput("26idcnpi:m17.VAL", yorigin, wait=True)
    if slowmotor == 'attoz':
        epics.caput("atto2:m3.VAL", xorigin, wait=True)
    elif slowmotor == 'attox':
        epics.caput("atto2:m4.VAL", xorigin, wait=True)

    print(" ")
    if len(hiccups):
        for hiccup in hiccups:
            print("number of bad points in line {0} : {1}".format(hiccup[0], hiccup[1]))

    xx, yy = np.meshgrid(abs_x, abs_y)
    flyxy_tail("samy", yy.T, slowmotor, abs_x, dwelltime, data_scaler)


###############################################################################################################


def flyxy_beam(dx0, dx1, nx, dy0, dy1, ny, dwelltime, delaytime=0.0, velofactor=0.92):

    data_scaler = np.zeros((2,ny,nx))

    flag_fluo, flag_eiger = get_detector_flags()

    if nx>124 and flag_fluo:
        sys.exit("number of points must be smaller than 124 for the fast axis...")
    if nx*ny > 16000:
        sys.exit("number of total points must be smaller than 16000...")
    fm_spd = np.abs(dx0-dx1)*1./nx/dwelltime
    if fm_spd < 0.1 or fm_spd > 5:
        sys.exit("hybridx motor speed must be between 0.1 and 5 um/s, currently {0:.1f}".format(fm_spd))

    while(epics.caget("PA:26ID:SCS_BLOCKING_BEAM.VAL") or epics.caget("PA:26ID:FES_BLOCKING_BEAM.VAL")):
        print("it seems that either the A or the C hutch shutter is closed, checking again in 1 minute")
        time.sleep(60)

    if not(epics.caget("26idc:1:userCalc5.VAL")): # and epics.caget("26idc:1:userCalc7.VAL")):
        sys.exit("please lock hybrid before performing this kind of scan")
    
    xorigin = epics.caget("26idcnpi:m34.VAL")
    xstart = dx0 + xorigin
    xend = dx1 + xorigin
    yorigin = epics.caget("26idcnpi:m35.VAL")
    ystart = dy0 + yorigin
    yend = dy1 + yorigin

    # epics.caput("26idcnpi:m35.VELO", 200, wait=True)
    epics.caput("26idcnpi:m35.VAL", ystart, wait=True)
    epics.caput("26idcnpi:m34.VAL", xstart, wait=True) 
    epics.caput("26idcnpi:m34.VELO", np.fabs(dx0-dx1)/nx/dwelltime*velofactor, wait=True)
    abs_y = np.linspace(ystart, yend, ny)
    abs_x = np.linspace(xstart, xend, nx)

    flyxy_head(ny, nx)

    hiccups = []

    print("scanning line"),

    def process1():
        time.sleep(delaytime+0.005) # before this got launched first
        ljm_fluo.send_fakedigital_singlebuffer(np.ones(nx), t1=dwelltime, out_num=0, dac_num=0, t0=0.005)
    def process2():
        time.sleep(delaytime)
        ljm_scaler.send_fakedigital_singlebuffer(np.ones(nx+1), t1=dwelltime, out_num=0, dac_num=0, t0=0.005, inverted=True)
        # need nx+1 to add one falling edge, draw the TTL diagram and you will understand why it is needed
    def process3():
        time.sleep(delaytime)
        ljm_mpx.send_fakedigital_singlebuffer(np.ones(nx), t1=dwelltime, out_num=0, dac_num=0, t0=0.01)
    def process4():
        time.sleep(delaytime)
        ljm_eiger.send_fakedigital_singlebuffer(np.ones(nx), t1=dwelltime, out_num=0, dac_num=0, t0=0.005) 

    for i_y in range(ny):

        epics.caput("26idc:3820:EraseStart",1)
        time.sleep(0.1)

        hiccup = 0
        print("{0:03d}/{1:03d}".format(i_y+1,ny)+"\b"*8),

        p1 = Process(target=process1)
        p4 = Process(target=process4)        
        p2 = Process(target=process2)        
        # using processes the delay between two processes is <5 ms
        # if sequentially launch one after another, it is >150 ms

        if flag_fluo:
            p1.start()
        if flag_eiger:
            p4.start()
        p2.start()
        
        epics.caput("26idcnpi:m34.VAL", xend, wait=True)

        #time.sleep(nx * dwelltime+.2)
        flag_break = False

        while(1):
            if flag_fluo:
                if epics.caget("26idcXMAP:Acquiring"):
                    hiccup += 1
                    print("waiting on fluo")
                    time.sleep(dwelltime)
                    flag_break = True
            if flag_eiger:
                num_cap = epics.caget("s26_eiger_cnm:HDF1:NumCaptured_RBV")
                if num_cap != (i_y+1)*nx:
                    hiccup += 1
                    print("waiting on eiger ", num_cap)
                    time.sleep(dwelltime)
                    flag_break = True
            if epics.caget("26idc:3820:Acquiring"):
                hiccup += 1
                print("waiting on scaler")
                time.sleep(dwelltime)
                flag_break = True
                if hiccup > 20:
                    print("stopping scaler")
                    epics.caput("26idc:3820:StopAll",1)
            if not flag_break:
                break
            flag_break = False

        epics.caput("26idcnpi:m35.VAL", abs_y[i_y])
        epics.caput("26idcnpi:m34.VELO", 200, wait=True)
        epics.caput("26idcnpi:m34.VAL", xstart, wait=True)
        epics.caput("26idcnpi:m34.VELO", np.fabs(dx0-dx1)/nx/dwelltime*velofactor, wait=True)
        #time.sleep(0.5) 

        if hiccup:
            hiccups+=[[i_y, hiccup]]

        for i_scaler in range(2):
            data_scaler[i_scaler, i_y] = epics.caget('26idc:3820:mca{0}'.format(i_scaler+2))[:nx]
        
        if flag_fluo:
            epics.caput("26idcXMAP:StartAll", 1)
            
        #time.sleep(0.5) # new for fei

    epics.caput("26idcnpi:m34.VELO", 200, wait=True)

    ljm_fluo.stop_streaming()
    ljm_scaler.stop_streaming()
    #ljm_mpx.stop_streaming()
    ljm_eiger.stop_streaming()

    epics.caput("26idcnpi:m35.VAL", yorigin)
    epics.caput("26idcnpi:m34.VAL", xorigin)

    print(" ")
    if len(hiccups):
        for hiccup in hiccups:
            print("number of bad points in line {0} : {1}".format(hiccup[0], hiccup[1]))

    xx, yy = np.meshgrid(abs_x, abs_y)
    flyxy_tail("hybridx", xx, "hybridy", abs_y, dwelltime, data_scaler)


#############################################################################################################################################

def flyxy_cleanup():

    flag_fluo, flag_eiger = get_detector_flags()

    ljm_fluo.stop_streaming()
    ljm_scaler.stop_streaming()
    ljm_mpx.stop_streaming()
    ljm_eiger.stop_streaming()

    epics.caput("26idcnpi:m17.VELO", 200, wait=True)
    epics.caput("26idcnpi:m34.VELO", 200, wait=True)

    epics.caput("26idc:filter:Fi1:Set",1)
    time.sleep(.1)

    epics.caput("s26_eiger_cnm:cam1:Acquire", 0)
    time.sleep(.1)
    epics.caput("s26_eiger_cnm:HDF1:Capture", 0) # End Capture
    time.sleep(.1)
    epics.caput("s26_eiger_cnm:Stats1:TS:TSAcquire", 0)
    time.sleep(.1)
    epics.caput("s26_eiger_cnm:Stats2:TS:TSAcquire", 0)
    time.sleep(.1)
    epics.caput("s26_eiger_cnm:Stats3:TS:TSAcquire", 0)
    time.sleep(.1)
    epics.caput("s26_eiger_cnm:Stats4:TS:TSAcquire", 0)
    time.sleep(.1)

    epics.caput("26idc:3820:StopAll", 1) # stop scaler acquisition
    time.sleep(.1)
    epics.caput("26idc:3820:ChannelAdvance", 0, wait=True) # set scaler to internal trigger
    epics.caput("26idc:3820:Channel1Source", 0, wait=True) 
    epics.caput("26idc:3820:PresetReal", 1, wait=True) 

    epics.caput("26idcXMAP:StopAll", 1) 
    time.sleep(.1)
    epics.caput("26idcXMAP:netCDF1:Capture", 0) # capture fluo
    time.sleep(.1)
    epics.caput("26idcXMAP:netCDF1:EnableCallbacks", 0, wait=True) # disable netcdf saving
    epics.caput("26idcXMAP:IgnoreGate", 1, wait=True) 
    epics.caput("26idcXMAP:CollectMode", 0, wait=True) # spectrum mapping

    epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 0, wait=True) # set fly flag = 0
    epics.caput("26idpvc:userCalc3.SCAN", 6, wait=True) # enable Martin's samy watchdog

    if epics.caget("26idc:1:userCalc7.VAL") or epics.caget("26idc:1:userCalc5.VAL"):
        if np.abs(epics.caget("26idcnpi:m35.RBV")-epics.caget("26idcnpi:Y_HYBRID_SP.VAL"))>0.5:
            print("returning hybrid motors to their original positions")
            epics.caput("26idcnpi:X_HYBRID_SP.VAL", epics.caget("26idcnpi:X_HYBRID_SP.VAL")+0.001)
            time.sleep(.1)
            epics.caput("26idcnpi:Y_HYBRID_SP.VAL", epics.caget("26idcnpi:Y_HYBRID_SP.VAL")+0.001)
            time.sleep(.1)
    
