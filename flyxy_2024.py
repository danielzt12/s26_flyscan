import sys
import epics
import epics.devices
import numpy as np
import time
import h5py
import netCDF4
import os
import fabio
sys.path.append("/home/sector26/pythonscripts/Tao/flyscan/ljm_scripts")
from ljm_functions import *
from readMDA import *

scanrecord = "26idbSOFT"

ljm_scaler = T7_ID26(2)
#ljm_fluo = T7_ID26(2) 
ljm_detector = T7_ID26(1) 
#ljm_eiger = T7_ID26(4)


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

    epics.caput("26idc:3820:ChannelAdvance", 1, wait=True) # set scaler to external trigger
    epics.caput("26idc:3820:Channel1Source", 1, wait=True) 
    epics.caput("26idc:3820:InputMode", 3, wait=True) # set scaler input mode to 3
    epics.caput("26idc:3820:NuseAll", pts_per_line, wait=True) # set channel to use to the nbpoints per line
    epics.caput("26idc:3820:PresetReal", 10000, wait=True) # set it to very long
    epics.caput("26idc:3820:scaler1.CONT", 0, wait=True)
    epics.caput("26idc:3820:scaler1.CNT", 0, wait=True) 
    
    if flag_fluo:
        epics.caput("26idcXMAP:CollectMode", 1, wait=True) #mca mapping
        epics.caput("26idcXMAP:PixelsPerRun", pts_per_line, wait=True) 
        epics.caput("26idcXMAP:IgnoreGate", 0, wait=True) #very important
        pathname_fluo = os.path.join(pathname, "fluo")
        if not os.path.isdir(pathname_fluo):
            os.mkdir(pathname_fluo)
            os.chmod(pathname_fluo, 0o777)
        pathname_fluo = "\\\\s26data\export"+pathname_fluo[5:]
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

    # ljm_scaler.write("DAC0", 3.3) # scaler is active low, need to set it to high at the beginning

    epics.caput("26idc:filter:Fi1:Set",0, wait=True)
    time.sleep(0.1)

    print ("completed in {0:.1f} sec".format(time.time()-t0))
    

def flyxy_tail(fastmotor, fastmotorvalues, slowmotor, slowmotorvalues, dwelltime, data_scaler):

    flag_fluo, flag_eiger = get_detector_flags()

    print ("returning config to normal...")

    pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)[:-4]

    epics.caput("26idc:filter:Fi1:Set",1)
    time.sleep(.1)

    epics.caput("26idc:3820:ChannelAdvance", 0, wait=True) # set scaler to internal trigger
    epics.caput("26idc:3820:Channel1Source", 0, wait=True) 
    epics.caput("26idc:3820:PresetReal", 1, wait=True) 

    for i in range(20): # wait 10 sec for file acquisition to complete
        if epics.caget("s26_eiger_cnm:HDF1:Capture_RBV"):
            time.sleep(0.5)
        else:
            break
    
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
        epics.caput("26idcXMAP:StopAll", 1) # ttt this should no longer be necessary since 20220225
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
    sr_grp.create_dataset("US Gap", data=epics.caget('S26ID:USID:GapM.VAL'))
    sr_grp.create_dataset("DS Gap", data=epics.caget('S26ID:DSID:GapM.VAL'))
    sr_grp.create_dataset("BPM H Position", data=epics.caget('S26:ID:SrcPt:HPositionM'))
    sr_grp.create_dataset("BPM V Position", data=epics.caget('S26:ID:SrcPt:VPositionM'))
    sr_grp.create_dataset("BPM H Angle", data=epics.caget('S26:ID:SrcPt:HAngleM'))
    sr_grp.create_dataset("BPM V Angle", data=epics.caget('S26:ID:SrcPt:VAngleM'))
    ida_grp = inst_grp.create_group("26-ID-A")
    for i in range(1,5):
        ida_grp.create_dataset(epics.caget('26idaWBS:m{0}.DESC'.format(i)), data=epics.caget('26idaWBS:m{0}.VAL'.format(i)))
    for i in range(1,8):
        ida_grp.create_dataset(epics.caget('26idaMIR:m{0}.DESC'.format(i)), data=epics.caget('26idaMIR:m{0}.VAL'.format(i)))  
    idb_grp = inst_grp.create_group("26-ID-B")
    for i in range(1,9):
        idb_grp.create_dataset(epics.caget('26idbDCM:sm{0}.DESC'.format(i)), data=epics.caget('26idbDCM:sm{0}.VAL'.format(i)))
    for i in range(1,5):
        idb_grp.create_dataset(epics.caget('26idbPBS:m{0}.DESC'.format(i)), data=epics.caget('26idbPBS:m{0}.VAL'.format(i)))
    idc_grp = inst_grp.create_group("26-ID-C")
    idc_grp.create_dataset("count_time", data=dwelltime)
    idc_grp.create_dataset(epics.caget('26idSAMTH:m1.DESC'.format(i)), data=epics.caget('26idSAMTH:m1.VAL'.format(i)))
    for i in range(1,7):
        idc_grp.create_dataset(epics.caget('npimic:m{0}.DESC'.format(i)), data=epics.caget('npimic:m{0}.VAL'.format(i)))
    for i in range(11,14):
        idc_grp.create_dataset(epics.caget('npimic:m{0}.DESC'.format(i)), data=epics.caget('npimic:m{0}.VAL'.format(i)))
    for i in range(21,25):
        idc_grp.create_dataset(epics.caget('npimic:m{0}.DESC'.format(i)), data=epics.caget('npimic:m{0}.VAL'.format(i)))
    for i in range(1,5):
        idc_grp.create_dataset(epics.caget('26idACS:m{0}.DESC'.format(i)), data=epics.caget('26idACS:m{0}.VAL'.format(i)))
    for i in range(1,4):
        idc_grp.create_dataset(epics.caget('26idMARS:m{0}.DESC'.format(i)), data=epics.caget('26idMARS:m{0}.VAL'.format(i)))
    for i in range(5,11):
        idc_grp.create_dataset(epics.caget('26idc:m{0}.DESC'.format(i)), data=epics.caget('26idc:m{0}.VAL'.format(i)))
    for i in range(13,15):
        idc_grp.create_dataset(epics.caget('26idc:m{0}.DESC'.format(i)), data=epics.caget('26idc:m{0}.VAL'.format(i)))
    idc_grp.create_dataset(epics.caget('26idcDET:m7.DESC'.format(i)), data=epics.caget('26idcDET:m7.VAL'.format(i)))
    idc_grp.create_dataset('NES H Slit', data=epics.caget('26idcNES:Slit1Hsize.VAL'))
    idc_grp.create_dataset('NES V Slit', data=epics.caget('26idcNES:Slit1Vsize.VAL'))
    for i in range(2,5):
        idc_grp.create_dataset(epics.caget('26idbSOFT:sm{0}.DESC'.format(i)), data=epics.caget('26idbSOFT:sm{0}.VAL'.format(i)))



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
    if slowmotor.lower() == "zpy":
        dim[1].p[0].name = "26idMARS:m3.VAL"
        dim[1].p[0].desc = 'ZPY'
    elif slowmotor.lower() == "zpx":
        dim[1].p[0].name = "26idMARS:m2.VAL"
        dim[1].p[0].desc = 'ZPX'
    elif slowmotor.lower() == "samth":
        dim[1].p[0].name = "26idSAMTH:m1.VAL"
        dim[1].p[0].desc = 'SAMTH'
    elif slowmotor.lower() == "zpz1":
        dim[1].p[0].name = "npimic:m21.VAL"
        dim[1].p[0].desc = 'ZPZ1'
    elif slowmotor.lower() == "zpz2":
        dim[1].p[0].name = "npimic:m22.VAL"
        dim[1].p[0].desc = 'ZPZ2'
    dim[1].p[0].data = slowmotorvalues

    dim[2].p.append(scanPositioner())
    dim[2].npts = nx
    dim[2].nd = 70
    dim[2].curr_pt = nx
    if fastmotor.lower() == "zpy":
        dim[2].p[0].name = "26idMARS:m3.VAL"
        dim[2].p[0].desc = 'ZPY'
    elif fastmotor.lower() == "zpx":
        dim[2].p[0].name = "26idMARS:m2.VAL"
        dim[2].p[0].desc = 'ZPX'
    dim[2].p[0].data = fastmotorvalues
    
    print("completed in {0:0.1f} sec".format(time.time()-t0))
    t0 = time.time()

    if flag_fluo:
        print("extracting fluo spectra from netcdf..."),
        fluonum =  epics.caget("26idcXMAP:netCDF1:FileNumber_RBV")-1
        f_fluo = os.path.join(pathname, "fluo",  "scan_{0}_{1:06d}.nc".format(scannum,fluonum))
        # os.chmod(f_fluo, 0o777) # disabled 2024 due to permission issue
        netcdffile = netCDF4.Dataset(f_fluo, "r")
        data_mca = np.swapaxes(netcdffile.variables['array_data'][:,:,256:].reshape(ny,2,124,256+2048*4)[:,:,:nx,256:].reshape(ny,2,nx,4,2048), 1,2).reshape(ny,nx,8,2048)
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
            dset[...] = data_mca[:,:,:-1,xmin:xmax+1].sum(2).sum(2)
        elif "mca" in name and flag_fluo:
            xmin = epics.caget(name+"LO")
            xmax = epics.caget(name+"HI")
            i_mod = int(name.split(".")[0][13:])-1
            dset.attrs["xmin"] = xmin
            dset.attrs["xmax"] = xmax
            dset.attrs["line"] = epics.caget(name+"NM")
            dset[...] = data_mca[:,:,i_mod,xmin:xmax+1].sum(2)
        elif "scaler1_cts1.B" in name:
            dset[...] = data_scaler[0]
        elif "scaler1_cts1.C" in name:
            dset[...] = data_scaler[1]
        elif "scaler1_cts1.D" in name:
            dset[...] = data_scaler[2]
        #elif "scaler1_cts2.A" in name:
        #     dset[...] = data_scaler[3]
        #elif "scaler1_cts2.B" in name:
        #    dset[...] = data_scaler[2]
            
        dim[2].d[i].data = dset[()]

    data_mca = None
    h5.close()
    print("completed in {0:0.1f} sec".format(time.time()-t0))

    f_mda = os.path.join(pathname, "mda", "26idbSOFT_{0:04d}.mda".format(scannum))

    writeMDA(dim, f_mda)
    print("{0} is now available.".format(f_mda))

    epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 0, wait=True) # set fly flag = 0

###############################################################################################################


def fly2d(motory, dy0, dy1, ny, motorx, dx0, dx1, nx, dwelltime, delaypts, waittime):

    t0 = time.time()

    data_scaler = np.zeros((3,ny,nx))

    flag_fluo, flag_eiger = get_detector_flags()

    if flag_fluo:
        if nx > 120 or nx < 40:
            sys.exit("put at least 40 and at most 120 points on the fast axis")

    if motorx.DESC != "ZPX" and motorx.DESC != "ZPY":
        sys.exit("need zpx or zpy on the fast axis")

    fm_spd = np.fabs(dx0-dx1)/nx/dwelltime
    print("piezo scanning speed: {0} um/s".format(fm_spd))

    if np.fabs(dx0-dx1)<=10:
        jb_spd = 5
        jb_set = 0.3
    elif np.fabs(dx0-dx1)<40:
        jb_spd = 20
        jb_set = 0.6
    else:
        jb_spd = 50
        jb_set = 1.5

    while(epics.caget("PA:26ID:SCS_BLOCKING_BEAM.VAL") or epics.caget("PA:26ID:FES_BLOCKING_BEAM.VAL")):
        print("it seems that either the A or the C hutch shutter is closed, checking again in 1 minute")
        time.sleep(60)

    if flag_eiger:
        if epics.caget("s26_eiger_cnm:cam1:Armed"):
            sys.exit("Eiger is still armed. Stop Eiger first.")
    
    xorigin = motorx.VAL
    xstart = xorigin + dx0
    xend = xorigin + dx1
    yorigin = motory.VAL
    ystart = dy0 + yorigin 
    yend = dy1 + yorigin

    if not epics.caget("26idMARS:m2.CNEN"):
        epics.caput("26idMARS:m2.CNEN", 1)
        print("unfreezing zpx")
        time.sleep(0.5)
    if not epics.caget("26idMARS:m3.CNEN"):
        epics.caput("26idMARS:m3.CNEN", 1)
        print("unfreezing zpy")
        time.sleep(0.5)
    if not epics.caget("26idMARS:m1.CNEN"):
        epics.caput("26idMARS:m1.CNEN", 1)
        print("unfreezing zpz")
        time.sleep(0.5)
    if not motory.CNEN:
        motory.CNEN = 1
        print("unfreezing ", motory.DESC)
        time.sleep(0.5)

    motorx.put("VAL", xstart, wait=False)
    motory.put("VAL", ystart, wait=True)
    motorx.put("VAL", xstart, wait=True) # both motors move at the same time but check x again 
    motorx.put("VELO", fm_spd, wait=True)
    abs_y = np.linspace(ystart, yend, ny)
    abs_x = np.linspace(xstart, xend, nx)

    flyxy_head(ny, nx)
    epics.caput("26idaWBS:sft01:ph01:ao07.VAL", xorigin, wait=True) 
    epics.caput("26idaWBS:sft01:ph01:ao08.VAL", yorigin, wait=True) 
    epics.caput("26idaWBS:sft01:ph01:ao11.VAL", nx, wait=True)
    epics.caput("26idaWBS:sft01:ph01:ao12.VAL", ny, wait=True)
    if motorx.DESC == "ZPX" and motory.DESC == "ZPY":
        epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 1, wait=True)
    elif motorx.DESC == "ZPY" and motory.DESC == "ZPX":
        epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 2, wait=True)
    elif motorx.DESC == "ZPX" and motory.DESC == "SAMTH":
        epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 3, wait=True)
    elif motorx.DESC == "ZPY" and motory.DESC == "SAMTH":
        epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 4, wait=True)
    else:
        epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 5, wait=True)
    
    hiccups = []

    ljm_detector.prepare_streaming()
    ljm_scaler.prepare_streaming()

    print("scanning line"),
    t1 = time.time()

    for i_y in range(ny):

        epics.caput("26idc:3820:EraseStart",1)
        time.sleep(0.1)

        hiccup = 0
        rt = (time.time()-t1)/i_y*(ny-i_y) if i_y > 0 else 0
        print("{0:03d}/{1:03d} remaining time {2:02d}m{3:02d}s".format(i_y+1,ny,int(rt/60), int(rt%60))+"\b"*30),
        sys.stdout.flush()
    
        ljm_detector.send_digital(nx, dwelltime, inverted=[0,0], delaypts=int(0.064/dwelltime) if delaypts <0 else delaypts)
        ljm_scaler.send_digital(nx+1, dwelltime, inverted=[1,0], delaypts=int(0.064/dwelltime) if delaypts <0 else delaypts)
        
        motorx.put("VAL", xend, wait=False)
        time.sleep((nx+1)*dwelltime) 

        flag_break = False

        while(1):
            if flag_fluo:
                if epics.caget("26idcXMAP:Acquiring"):
                    hiccup += 1
                    #print("waiting on fluo", epics.caget("26idcXMAP:dxp1:CurrentPixel"))
                    flag_break = True
                    if hiccup > 20 and epics.caget("26idcXMAP:dxp1:CurrentPixel") == nx:
                        print("stopping fluo")
                        epics.caput("26idcXMAP:StopAll", 1) 
            if flag_eiger:
                num_cap = epics.caget("s26_eiger_cnm:cam1:NumImagesCounter_RBV") #####s26_eiger_cnm:HDF1:NumCaptured_RBV")
                if num_cap != (i_y+1)*nx:
                    hiccup += 1
                    #print("waiting on eiger ", num_cap)
                    time.sleep(dwelltime)
                    flag_break = True
                        
            if epics.caget("26idc:3820:Acquiring"):
                hiccup += 1
                #print("waiting on scaler")
                time.sleep(dwelltime)
                flag_break = True
                if hiccup > 20:
                    print("stopping scaler")
                    epics.caput("26idc:3820:StopAll",1)
            if not flag_break:
                break
            flag_break = False
            time.sleep(dwelltime)
            
        if i_y+1 < ny:
            motory.put("VAL", abs_y[i_y+1])
            motorx.put("VELO", jb_spd, wait=True)
            motorx.put("VAL", xstart, wait=True)
            motorx.put("VELO", fm_spd, wait=True)
            time.sleep(jb_set + waittime) # positioner settling time
            if flag_fluo:
                epics.caput("26idcXMAP:StartAll", 1)

        if hiccup:
            hiccups+=[[i_y, hiccup]]

        try:
            data_scaler[0, i_y] = epics.caget('26idc:3820:mca2')[:nx]
            data_scaler[1, i_y] = epics.caget('26idc:3820:mca3')[:nx]
            data_scaler[2, i_y] = epics.caget('26idc:3820:mca4')[:nx]
        except TypeError:
            pass

        if epics.caget("26idaWBS:sft01:ph01:ao06.VAL") < 0:
            abs_y = abs_y[:(i_y+1)]
            data_scaler = data_scaler[:,:(i_y+1)]
            break


    motorx.put("VELO", 5, wait=True)

    ljm_detector.stop_streaming(NumAddr=1) # only one addr was used, so clean up just the one
    ljm_scaler.stop_streaming(NumAddr=1) # only one addr was used, so clean up just the one

    motory.put("VAL", yorigin, wait=False)
    motorx.put("VAL", xorigin, wait=True)
    motory.put("VAL", yorigin, wait=True)

    print(" ")
    #if len(hiccups):
    #    for hiccup in hiccups:
    #        print("number of bad points in line {0} : {1}".format(hiccup[0], hiccup[1]))

    #xend = dx1 + xorigin # why are these two lines even here
    #abs_x = np.linspace(xstart, xend, nx)  # why are these two lines even here
    xx, yy = np.meshgrid(abs_x, abs_y)
    flyxy_tail(motorx.DESC, xx, motory.DESC, abs_y, dwelltime, data_scaler)

    print("elaspsed time: {0:.0f} sec".format(time.time()-t0))


     
#############################################################################################################################################

def fly_cleanup():

    flag_fluo, flag_eiger = get_detector_flags()

    ljm_detector.stop_streaming(NumAddr=128) # clean up all possible addr
    ljm_scaler.stop_streaming(NumAddr=128) # clean up all possible addr

    #epics.caput("26idACS:m2.VELO", 200, wait=True)
    epics.caput("26idMARS:m2.VELO", 5, wait=True)
    epics.caput("26idMARS:m3.VELO", 5, wait=True)

    epics.caput("26idc:filter:Fi1:Set",1)
    time.sleep(.1)

    epics.caput("26idc:3820:StopAll", 1) # stop scaler acquisition
    time.sleep(.1)
    epics.caput("26idc:3820:ChannelAdvance", 0, wait=True) # set scaler to internal trigger
    epics.caput("26idc:3820:Channel1Source", 0, wait=True) 
    epics.caput("26idc:3820:PresetReal", 1, wait=True) 

    if flag_fluo:
        epics.caput("26idcXMAP:StopAll", 1) 
        time.sleep(.1)
        epics.caput("26idcXMAP:netCDF1:Capture", 0) # capture fluo
        time.sleep(.1)
        epics.caput("26idcXMAP:netCDF1:EnableCallbacks", 0, wait=True) # disable netcdf saving
        epics.caput("26idcXMAP:IgnoreGate", 1, wait=True) 
        epics.caput("26idcXMAP:CollectMode", 0, wait=True) # spectrum mapping

    
    if flag_eiger:
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
    
    epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 0, wait=True) # set fly flag = 0

    
"""
def flyxy_beam(dx0, dx1, nx, dy0, dy1, ny, dwelltime, delaypts, waittime):

    t0 = time.time()

    data_scaler = np.zeros((3,ny,nx))

    flag_fluo, flag_eiger = get_detector_flags()

    if flag_fluo:
        if nx > 120 or nx < 40:
            sys.exit("put at least 40 and at most 120 points on the fast axis (zpx)")

    fm_spd = np.fabs(dx0-dx1)/nx/dwelltime
    print("piezo scanning speed: {0} um/s".format(fm_spd))

    if np.fabs(dx0-dx1)<=10:
        jb_spd = 5
        jb_set = 0.3
    elif np.fabs(dx0-dx1)<40:
        jb_spd = 20
        jb_set = 0.6
    else:
        jb_spd = 50
        jb_set = 1.5

    #if fm_spd < 0.01 or fm_spd > 10:
    #    sys.exit("hybridx motor speed must be between 0.01 and 10 um/s, currently {0:.1f}".format(fm_spd))

    while(epics.caget("PA:26ID:SCS_BLOCKING_BEAM.VAL") or epics.caget("PA:26ID:FES_BLOCKING_BEAM.VAL")):
        print("it seems that either the A or the C hutch shutter is closed, checking again in 1 minute")
        time.sleep(60)

    if flag_eiger:
        if epics.caget("s26_eiger_cnm:cam1:Armed"):
            sys.exit("Eiger is still armed. Stop Eiger first.")
    
    xorigin = epics.caget("26idMARS:m2.VAL")
    xstart = dx0 + xorigin
    xend = dx0 + xorigin +(dx1-dx0) # /((1+velofactor)/2) # I commented this in 2024, but it might be a mistake
    yorigin = epics.caget("26idMARS:m3.VAL")
    ystart = dy0 + yorigin 
    yend = dy1 + yorigin

    if not epics.caget("26idMARS:m2.CNEN"):
        epics.caput("26idMARS:m2.CNEN", 1)
        print("unfreezing zpx")
        time.sleep(0.5)
    if not epics.caget("26idMARS:m3.CNEN"):
        epics.caput("26idMARS:m3.CNEN", 1)
        print("unfreezing zpy")
        time.sleep(0.5)
    if not epics.caget("26idMARS:m1.CNEN"):
        epics.caput("26idMARS:m1.CNEN", 1)
        print("unfreezing zpz")
        time.sleep(0.5)

    if epics.caget("npimic:m23.CNEN"):
        epics.caput("npimic:m23.CNEN", 0)
        print("freezing samx")
        time.sleep(0.5)
    if epics.caget("26idACS:m2.CNEN"):
        epics.caput("26idACS:m2.CNEN", 0)
        print("freezing samy")
        time.sleep(0.5)
    if epics.caget("npimic:m24.CNEN"):
        epics.caput("npimic:m24.CNEN", 0)
        print("freezing samz")
        time.sleep(0.5)

    # epics.caput("26idMARS:m3.VELO", 200, wait=True)
    epics.caput("26idMARS:m3.VAL", ystart, wait=True)
    epics.caput("26idMARS:m2.VAL", xstart, wait=True)
    epics.caput("26idMARS:m2.VELO", fm_spd, wait=True) 
    abs_y = np.linspace(ystart, yend, ny)
    abs_x = np.linspace(xstart, xend, nx)

    flyxy_head(ny, nx)
    epics.caput("26idaWBS:sft01:ph01:ao07.VAL", xorigin, wait=True)
    epics.caput("26idaWBS:sft01:ph01:ao08.VAL", yorigin, wait=True)
    epics.caput("26idaWBS:sft01:ph01:ao11.VAL", nx, wait=True)
    epics.caput("26idaWBS:sft01:ph01:ao12.VAL", ny, wait=True)
    epics.caput("26idaWBS:sft01:ph01:ao06.VAL", 1, wait=True) # set fly flag = 1
    
    hiccups = []

    ljm_detector.prepare_streaming()
    ljm_scaler.prepare_streaming()

    print("scanning line"),
    t1 = time.time()

    for i_y in range(ny):

        epics.caput("26idc:3820:EraseStart",1)
        time.sleep(0.1)

        hiccup = 0
        rt = (time.time()-t1)/i_y*(ny-i_y) if i_y > 0 else 0
        print("{0:03d}/{1:03d} remaining time {2:02d}m{3:02d}s".format(i_y+1,ny,int(rt/60), int(rt%60))+"\b"*30),
        sys.stdout.flush()
    
        ljm_detector.send_digital(nx, dwelltime, inverted=[0,0], delaypts=int(0.064/dwelltime) if delaypts <0 else delaypts)
        ljm_scaler.send_digital(nx+1, dwelltime, inverted=[1,0], delaypts=int(0.064/dwelltime) if delaypts <0 else delaypts)
        
        epics.caput("26idMARS:m2.VAL", xend, wait=False)
        time.sleep((nx+1)*dwelltime) 

        flag_break = False

        while(1):
            if flag_fluo:
                if epics.caget("26idcXMAP:Acquiring"):
                    hiccup += 1
                    #print("waiting on fluo", epics.caget("26idcXMAP:dxp1:CurrentPixel"))
                    flag_break = True
                    if hiccup > 20 and epics.caget("26idcXMAP:dxp1:CurrentPixel") == nx:
                        print("stopping fluo")
                        epics.caput("26idcXMAP:StopAll", 1) 
            if flag_eiger:
                num_cap = epics.caget("s26_eiger_cnm:cam1:NumImagesCounter_RBV") #####s26_eiger_cnm:HDF1:NumCaptured_RBV")
                if num_cap != (i_y+1)*nx:
                    hiccup += 1
                    #print("waiting on eiger ", num_cap)
                    time.sleep(dwelltime)
                    flag_break = True
                        
            if epics.caget("26idc:3820:Acquiring"):
                hiccup += 1
                #print("waiting on scaler")
                time.sleep(dwelltime)
                flag_break = True
                if hiccup > 20:
                    print("stopping scaler")
                    epics.caput("26idc:3820:StopAll",1)
            if not flag_break:
                break
            flag_break = False
            time.sleep(dwelltime)
            
        if i_y+1 < ny:
            epics.caput("26idMARS:m3.VAL", abs_y[i_y+1])
            epics.caput("26idMARS:m2.VELO", jb_spd, wait=True)
            epics.caput("26idMARS:m2.VAL", xstart, wait=True)
            epics.caput("26idMARS:m2.VELO", fm_spd, wait=True)
            time.sleep(jb_set + waittime) # positioner settling time
            if flag_fluo:
                epics.caput("26idcXMAP:StartAll", 1)

        if hiccup:
            hiccups+=[[i_y, hiccup]]

        try:
            data_scaler[0, i_y] = epics.caget('26idc:3820:mca2')[:nx]
        except TypeError:
            pass
        try:
            data_scaler[1, i_y] = epics.caget('26idc:3820:mca3')[:nx]
        except TypeError:
            pass
        try:
            data_scaler[2, i_y] = epics.caget('26idc:3820:mca4')[:nx]
        except TypeError:
            pass
        #try:
        #    data_scaler[3, i_y] = epics.caget('26idc:3820:mca5')[:nx]
        #except TypeError:
        #    pass

    epics.caput("26idMARS:m2.VELO", 5, wait=True)

    ljm_detector.stop_streaming(NumAddr=1) # only one addr was used, so clean up just the one
    ljm_scaler.stop_streaming(NumAddr=1) # only one addr was used, so clean up just the one

    epics.caput("26idMARS:m3.VAL", yorigin)
    epics.caput("26idMARS:m2.VAL", xorigin)

    print(" ")
    #if len(hiccups):
    #    for hiccup in hiccups:
    #        print("number of bad points in line {0} : {1}".format(hiccup[0], hiccup[1]))

    xend = dx1 + xorigin
    abs_x = np.linspace(xstart, xend, nx)
    xx, yy = np.meshgrid(abs_x, abs_y)
    flyxy_tail("zpx", xx, "zpy", abs_y, dwelltime, data_scaler)

    print("elaspsed time: {0:.0f} sec".format(time.time()-t0))

"""
