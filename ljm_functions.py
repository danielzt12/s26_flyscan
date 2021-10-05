from labjack import ljm
import time
import numpy as np
from fractions import gcd

ljm.loadConstantsFromFile("ljm_constants.json")

def listallT7():

    result = ljm.listAll(7,3)
    print("found {0} devices".format(result[0]))
    for i in range(result[0]):
        print ("device {0} : SN [{1}] IP [{2}]".format(i, result[3][i], ljm.numberToIP(result[4][i])))


class T7_ID26:


    def fprintf(self, ports, ScanRate, filename, max_length = 1000, ScansPerRead = 1, verbose = False):

        self.stop_streaming()
        if not isinstance(ports, (list,)):
            ports = [ports]
        NumAddr = len(ports)
        Addr = []
        for port in ports:
            Addr += [ljm.nameToAddress(port)[0]]
        output_file = open(filename, "w")
        ljm.eStreamStart(self.handle, ScansPerRead, NumAddr, Addr, ScanRate)
        while(max_length): # which means if you put a negative value, it will run forever
            newline = ljm.eStreamRead(self.handle)
            output_file.write(" ".join(str(el) for el in newline[0])+"\n")
            if verbose:
                print max_length, newline
            max_length -= 1
        ljm.eStreamStop(self.handle)
        output_file.close()
            

    def start_streaming(self, ports):

        if not isinstance(ports, (list,)):
            ports = [ports]
        NumAddr = len(ports)
        Addr = []
        for port in ports:
            Addr += [ljm.nameToAddress(port)[0]]
        
        for i in range(NumAddr):
            r = self.write("STREAM_SCANLIST_ADDRESS{0}".format(i), Addr[i])
            # print Addr[i], r
            # print("writing to scan list [{0}]...{1} ".format(i, r == Addr[i]))
        r = self.write("STREAM_NUM_ADDRESSES", NumAddr)
        # print("writing number of address = {0}".format(r))
        r = self.write("STREAM_ENABLE", 1)
        # print("stream enabled...{0}".format(r==1))
            

    def stop_streaming(self):

        r = self.read("STREAM_ENABLE")
        if r == 0:
            pass
            # print "stream was not running"
        else:
            r = self.write("STREAM_ENABLE", 0)
            # print "disabling stream just in case...", r==0
        # print "clean up the scan list just in case"
        for i in range(128):
            if self.write("STREAM_SCANLIST_ADDRESS{0}".format(i), 0) != 0:
                # print "problem!!!"
                pass
        #print "setting DACs to 0..."
        #self.write("DAC0", 0)
        #self.write("DAC1", 0)
    

    def send_digital(self, data, t1, buffersize=4096):

        t0 = 0.01
        ScanRate = 2e3/(gcd(t0*1000,t1*1000))
        if ScanRate > 1000:
            # print "ScanRate too high :", ScanRate
            return 0
        if len(data.shape) == 1:
            data = (data[np.newaxis, :]>0)
        self.stop_streaming()
        r = self.write("STREAM_SCANRATE_HZ", ScanRate)
        # print("setting scan rate to {0}".format(r))
        data = (data * np.arange(data.shape[0])[:,np.newaxis]).astype(int)
        # print data.dtype
        data = np.left_shift(1,data).sum(0)
        data = (np.array([1]*int((t1-t0)*ScanRate)+[0]*int(t0*ScanRate)) * np.c_[data]).flatten()
        data = data * np.ones(2,dtype=int).reshape(2,1)
        # print data
        data_inhib = 0x7FFFFF - data
        # print "Total data to be transmitted: {0}x{1} at {2} Hz".format(data.shape[0], data.shape[1], ScanRate)
        self.map_stream_output(0, "FIO_STATE", buffersize=buffersize)
        self.map_stream_output(1, "FIO_DIRECTION", buffersize=buffersize)
        #self.map_stream_output(2, "DIO_INHIBIT", buffersize=buffersize)
        self.start_streaming(["STREAM_OUT0", "STREAM_OUT1"])
        self.data_to_buffer([0, 1], data, buffertype = "U16")
        self.stop_streaming()


    def send_fakedigital_ringbuffer(self, data_raw, t1, t0 = None, out_num = 0, dac_num = 0, buffersize = 4096, vmax = 3.3):
        
        if t0 == None:
            t0 = t1 - 0.005 # up time has been fixed to 5 ms
        ScanRate = 1000 # ScanRate has been fixed to 1kHz
        #ScanRate = 5e3/(gcd(t0*1000,t1*1000))
        if not isinstance(out_num, (list,)):
            out_num = [out_num]; dac_num = [dac_num]; data_raw = data_raw[np.newaxis,:]
        envelop = np.array([1]*int((t1-t0)*ScanRate)+[0]*int(t0*ScanRate))
        data = np.zeros((data_raw.shape[0], data_raw.shape[1]*envelop.shape[0]))
        for i in range(data.shape[0]):
            data[i] = (envelop * np.c_[data_raw[i]>0]).flatten()*vmax
        self.send_analog_ringbuffer(data, ScanRate, out_num, dac_num, buffersize)


    def send_analog_ringbuffer(self, data, ScanRate, out_num = 0, dac_num = 0, buffersize = 4096):

        if not isinstance(out_num, (list,)):
            out_num = [out_num]; dac_num = [dac_num]; data = data[np.newaxis,:]
        self.stop_streaming()
        # print "Total data to be transmitted: {0}x{1} at {2} Hz".format(data.shape[0], data.shape[1], ScanRate)
        r = self.write("STREAM_SCANRATE_HZ", ScanRate)
        # print("setting scan rate to {0}".format(r))
        for i in range(data.shape[0]):
            self.map_stream_output(out_num[i], "DAC{0}".format(dac_num[i]), buffersize=buffersize)
        self.start_streaming(" ".join(map("STREAM_OUT{0}".format, out_num)).split())
        self.data_to_ringbuffer(out_num, data, buffertype = "F32")


    def send_fakedigital_singlebuffer(self, data_raw, t1, t0 = 0.005, out_num = 0, dac_num = 0, vmax = 3.3, inverted = False):
        
        n_data = data_raw.shape[0]
        n_total = int(16382/2/n_data)
        n_down = max(int(round(t0/t1*n_total,0)), 1)
        n_up = int(n_total - n_down)
        ScanRate = 1/t1*n_total # trigger at rise edge
        data = (np.ones((n_data,n_total))*np.array([1]*n_up+[0]*n_down)).flatten()
        data = np.append(np.zeros(1), data)
        data = np.append(data, np.zeros(16382/2-n_total*n_data))*vmax
        if inverted:
            data = vmax - data
        self.send_analog_singlebuffer(data, ScanRate, out_num, dac_num)


    def send_analog_singlebuffer(self, data, ScanRate, out_num = 0, dac_num = 0):

        self.stop_streaming()
        r = self.write("STREAM_SCANRATE_HZ", ScanRate)
        #print("setting scan rate to {0}".format(r))
        self.map_stream_output(out_num, "DAC{0}".format(dac_num), buffersize=16384)
        self.start_streaming("STREAM_OUT{0}".format(out_num))
        self.data_to_singlebuffer(out_num, data, buffertype = "F32")


    def data_to_singlebuffer(self, out_num, data, buffertype = "F32"):

        bytesperdata = int(buffertype[1:])/16
        buffersize = int(self.read("STREAM_OUT{0}_BUFFER_ALLOCATE_NUM_BYTES".format(out_num)))
        # print("buffersize on OUTPUT{0} is {1} bytes".format(out_num, buffersize))
        r = self.read("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num))
        if r != 0:
            # print("loop size is non-zero, setting it to zero")
            r = self.write("STREAM_OUT{0}_ENABLE".format(out_num), 0)
            # print("Disabling output{0}...{1}".format(out_num, r==0))
            r = self.write("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num), 0)
            r = self.write("STREAM_OUT{0}_ENABLE".format(out_num), 1)
            # print("Enabling output{0}...{1}".format(out_num, r==1))
        # print("Writing data into buffer") # very important ! has to be dne after setting nloop
        self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num, buffertype), data)
        self.write("STREAM_OUT{0}_SET_LOOP".format(out_num), 1, writeonly = True)
        # print("Transferring data immediately") # will be replaced by trigger
        
          


    def data_to_ringbuffer(self, out_num, data, buffertype = "F32"):

        bytesperdata = int(buffertype[1:])/16
        buffersize = int(self.read("STREAM_OUT{0}_BUFFER_ALLOCATE_NUM_BYTES".format(out_num[0])))
        # print("buffersize on OUTPUT{0} is {1} bytes".format(out_num[0], buffersize))
        # print("yourdata in {0} is {1} bytes".format(buffertype, data.shape[1]*bytesperdata))
        if buffersize > data.shape[1]*bytesperdata:
            for i in range(data.shape[0]):
                # print("Infinite loop over existing data")
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 0)
                # print("Disabling output{0}...{1}".format(out_num[i], r==0))
                r = self.write("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]), data.shape[1])
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 1)
                # print("Enabling output{0}...{1}".format(out_num[i], r==1))
                r = self.read("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]))
                # print("setting loop size to {0} values / {1} bytes".format(r, r*bytesperdata))
                # print("Writing data into buffer") # very important ! has to be dne after setting nloop
                self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), data[i])
                self.write("STREAM_OUT{0}_SET_LOOP".format(out_num[i]), 2, writeonly = True)
                # print("Transferring data immediately") # will be replaced by trigger
        else:
            if data.shape[1]%(buffersize/4):
                ndata = (data.shape[1]/(buffersize/4)+1) * (buffersize/4)
                # print("appending data with zeros")
                data = np.append(data, np.zeros((data.shape[0],ndata-data.shape[1])), axis=1)
            else:
                ndata = data.shape[1]
            data = data.reshape(data.shape[0], ndata/(buffersize/4), buffersize/4)
            # print("stream will loop {0} times with size of {1}".format(data.shape[1], data.shape[2]))
            for i in range(data.shape[0]):
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 1)
                self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), np.zeros(buffersize/2))
                self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), data[i,0])
                # this is important, if not disabled, we cannot affect loop size!
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 0)
                # print("Disabling output{0}...{1}".format(out_num[i], r==0))
                r = self.write("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]), buffersize/4)
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 1)
                # print("Enabling output{0}...{1}".format(out_num[i], r==1))
                r = self.read("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]))
                # print("setting loop size to {0} values / {1} bytes".format(r, r*bytesperdata))
                # print("Writing initial data into half of the buffer")
                self.write("STREAM_OUT{0}_SET_LOOP".format(out_num[i]), 1, writeonly = True)
                # print("Transferring data immediately") # will be replaced by trigger
            scanrate = self.read("STREAM_SCANRATE_HZ")
            check_interval = 1/scanrate 
            time.sleep(check_interval*10)
            for j in range(1, data.shape[1]):
                emptybuffer = self.read("STREAM_OUT{0}_BUFFER_STATUS".format(out_num[0]))
                while emptybuffer*2 <buffersize/2:
                    emptybuffer = self.read("STREAM_OUT{0}_BUFFER_STATUS".format(out_num[0]))
                    nn = int(50*emptybuffer*2/buffersize)
                    #print("\r{0:05d}/{1:05d} [".format(j,data.shape[1])+"|"*nn+" "*(50-nn)+"]"),
                    time.sleep(check_interval)
                # print("\rWriting data into buffer"),
                for i in range(data.shape[0]):
                    self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), data[i,j])
                    self.write("STREAM_OUT{0}_SET_LOOP".format(out_num[i]), 1, writeonly = True)
                time.sleep(check_interval*10)
            self.stop_streaming()

    def data_streamout(self):

        self.write("STREAM_OUT3_SET_LOOP", 3, writeonly = True)
        

    def map_stream_output(self, out_num, target, buffersize=4096):

        r = self.write("STREAM_OUT{0}_ENABLE".format(out_num), 0)
        # print("Disabling output{0}...{1}".format(out_num, r==0))
        target_addr = ljm.nameToAddress(target)[0]
        r = self.write("STREAM_OUT{0}_TARGET".format(out_num), target_addr)
        # print("Setting output{0} to {1}...{2}".format(out_num, target, r==target_addr))
        r = self.write("STREAM_OUT{0}_BUFFER_ALLOCATE_NUM_BYTES".format(out_num), buffersize)
        # print("Setting buffersize to {0}".format(r))
        r = self.write("STREAM_OUT{0}_ENABLE".format(out_num), 1)
        # print("Enabling output{0}...{1}".format(out_num, r==1))
               

    def writearray(self, name, array):

        addr, typ = ljm.nameToAddress(name)
        ljm.eWriteAddressArray(self.handle, addr, typ, len(array), array)

    def write(self, name, value, writeonly = False):
            
        ljm.eWriteName(self.handle, name, value)
        if not writeonly:
            return ljm.eReadName(self.handle, name)

    def read(self, name):
              
        return ljm.eReadName(self.handle, name)
              
    def close(self):
        
        ljm.close(self.handle)
    

    def __init__(self, IP):

        if isinstance(IP, int):
            IP = "164.54.128.10{0}".format(IP)
        try:
            self.handle = ljm.open(7,3,IP)
        except Exception as e:
            print e
            print "Accepted variables are IPs or integers 1 2 3 4"
            print "type listallT7() to list the IPs"
        info = ljm.getHandleInfo(self.handle)
        print("T7 device connected")
        print("SN : {0}".format(info[2]))
        print("IP : {0}".format(ljm.numberToIP(info[3])))
        print("Port : {0}".format(info[4]))
        print("MaxBytesPerMB : {0}".format(info[5]))



"""
#r = self.write("STREAM_SAMPLES_PER_PACKET", 1)
        #print("setting sample per package to {0}".format(r)) # this has no effect anyway
        #r = self.write("STREAM_TRIGGER_INDEX", 0)
        #print("setting streaming scanning to start when stream is enabled...{0}".format(r==0))  
        #r = self.write("STREAM_CLOCK_SOURCE", 0)
        #print("setting clock source to internal crystal...{0}".format(r==0))  



   def data_to_buffer(self, out_num, data, buffertype = "F32"):

        bytesperdata = int(buffertype[1:])/16
        buffersize = int(self.read("STREAM_OUT{0}_BUFFER_ALLOCATE_NUM_BYTES".format(out_num[0])))
        print("buffersize on OUTPUT{0} is {1} bytes".format(out_num[0], buffersize))
        print("yourdata in {0} is {1} bytes".format(buffertype, data.shape[1]*bytesperdata))
        if buffersize > data.shape[1]*bytesperdata:
            for i in range(data.shape[0]):
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 0)
                print("Disabling output{0}...{1}".format(out_num[i], r==0))
                r = self.write("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]), data.shape[1])
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 1)
                print("Enabling output{0}...{1}".format(out_num[i], r==1))
                r = self.read("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]))
                print("setting loop size to {0} values / {1} bytes".format(r, r*bytesperdata))
                print("Writing data into buffer") # very important ! has to be dne after setting nloop
                self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), data[i])
                self.write("STREAM_OUT{0}_SET_LOOP".format(out_num[i]), 1, writeonly = True)
                print("Transferring data immediately") # will be replaced by trigger
        else:
            if data.shape[1]%(buffersize/8):
                ndata = (data.shape[1]/(buffersize/8)+1) * (buffersize/8)
                print("appending data with zeros")
                data = np.append(data, np.zeros((data.shape[0],ndata-data.shape[1])), axis=1)
            else:
                ndata = data.shape[1]
            data = data.reshape(data.shape[0], ndata/(buffersize/8), buffersize/8)
            print("stream will loop {0} times with size of {1}".format(data.shape[1], data.shape[2]))
            for i in range(data.shape[0]):
                # this is important, if not disabled, we cannot affect loop size!
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 0)
                print("Disabling output{0}...{1}".format(out_num[i], r==0))
                r = self.write("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]), buffersize/8)
                r = self.write("STREAM_OUT{0}_ENABLE".format(out_num[i]), 1)
                print("Enabling output{0}...{1}".format(out_num[i], r==1))
                r = self.read("STREAM_OUT{0}_LOOP_NUM_VALUES".format(out_num[i]))
                print("setting loop size to {0} values / {1} bytes".format(r, r*2))
                print("Writing initial data into half of the buffer")
                self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), data[i,:2].flatten())
                self.write("STREAM_OUT{0}_SET_LOOP".format(out_num[i]), 1, writeonly = True)
                print("Transferring data immediately") # will be replaced by trigger
            scanrate = self.read("STREAM_SCANRATE_HZ")
            check_interval = 1/scanrate #buffersize/8/scanrate
            time.sleep(check_interval*10)
            for j in range(2, data.shape[1]):
                emptybuffer = self.read("STREAM_OUT{0}_BUFFER_STATUS".format(out_num[0]))
                while emptybuffer*2 <buffersize/2:
                    emptybuffer = self.read("STREAM_OUT{0}_BUFFER_STATUS".format(out_num[0]))
                    nn = int(50*emptybuffer*2/buffersize)
                    print("\r{0:05d}/{1:05d} [".format(j,data.shape[1])+"|"*nn+" "*(50-nn)+"]"),
                    time.sleep(check_interval)
                print("\rWriting data into buffer"),
                for i in range(data.shape[0]):
                    self.writearray("STREAM_OUT{0}_BUFFER_{1}".format(out_num[i], buffertype), data[i,j])
                    self.write("STREAM_OUT{0}_SET_LOOP".format(out_num[i]), 1, writeonly = True)
                time.sleep(check_interval*10)
            self.stop_streaming()
                    




"""
