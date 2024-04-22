import struct
import numpy as np
import datetime as datetime

_Raw8_Fields = [("version","5s"),
    ("numSpectra","B"),
    ("length","I"),
    ("seqNum","B"),
    ("measMode","B",{0:"scope",1:"absorbance",2:"scope corrected for dark",3:"transmission",4:"reflectance",5:"irradiance",6:"relative irradiance",7:"temperature"}),
    ("bitness","B"),
    ("SDmarker","B"),
    ("specID","10s"),
    ("userfriendlyname","64s"),
    ("status","B"),
    ("startPixel","H"),
    ("stopPixel","H"),
    ("IntTime","f"),
    ("integrationdelay","I"),
    ("Avg","I"),
    ("enable","B"),
    ("forgetPercentage","B"),
    ("Boxcar","H"),
    ("smoothmodel","B"),
    ("saturationdetection","B"),
    ("TrigMode","B"),
    ("TrigSource","B"),
    ("TrigSourceType","B"),
    ("strobeCtrl","H"),
    ("laserDelay","I"),
    ("laserWidth","I"),
    ("laserWavelength","f"),
    ("store2ram","H"),
    ("timestamp","I"),
    ("SPCfiledate","I"),
    ("detectorTemp","f"),
    ("boardTemp","f"),
    ("NTC2volt","f"),
    ("ColorTemp","f"),
    ("CalIntTime","f"),
    ("fitdata","5d"),
    ("comment","130s")
]

def PlanckFunction(temp: float, wl: np.array):
    c = 3e8 # speed of light [m s**−1]
    h = 6.625e-34 # Planck constant [J s]
    kb = 1.38e-23 # Boltzmann constant [J K**−1]
    return ((2*h*c*c)/(wl**5)) * 1./(np.exp((h*c)/(wl*kb*temp))-1)

class Raw8:
    def __init__(self, filename: str):
        self.header = {}
        with open(filename,"rb") as f:
            for k in _Raw8_Fields:
                s = struct.Struct(k[1])
                dat = s.unpack(f.read(s.size))
                if len(dat)==1:
                    dat = dat[0]
                self.header[k[0]] = dat
            dataLength =  self.header['stopPixel']-self.header['startPixel']+1
            self.dataLenth = dataLength
            self.data = {
                'wl': struct.unpack(f"<{dataLength}f",f.read(4*dataLength)),
                'scope': struct.unpack(f"<{dataLength}f",f.read(4*dataLength)),
                'dark': struct.unpack(f"<{dataLength}f",f.read(4*dataLength)),
                'ref': struct.unpack(f"<{dataLength}f",f.read(4*dataLength))
                }
    
    def getData(self, name: str):
        return np.array(self.data[name])

    def getScope(self):
        return self.getData("scope")

    def getWavelength(self):
        return self.getData("wl")

    def getDark(self):
        return self.getData("dark")

    def getRef(self):
        return self.getData("ref")

    def getBlackBody(self):
        return PlanckFunction(self.header['ColorTemp'], self.getWavelength()*1e-9)

    def getRelativeIrradiance(self):
        return self.getBlackBody()*(self.getScope()-self.getDark())
# This is from the manual: http://www.content.mphotonics.de/AVA/AVASOFT_Manual_8.4.pdf
# Page 35. But I'm not conviced that this is correct

    def getDate(self):
        d = self.header['SPCfiledate']
        return {'year':d>>20,
            'month': (d>>16)%(2**4),
            'day': (d>>11)%(2**5),
            'hour': (d>>6)%(2**5),
            'minute': d%(2**6)}

    def getDatetime(self):
        d = self.getDate()
        return datetime.datetime(d['year'],d['month'],d['day'],d['hour'],d['minute'])