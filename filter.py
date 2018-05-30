import numpy as np
import copy
import globals as G

class StationFilter:

    def __init__(self, data):
        self.data = data


    def filter_according_tag(self, tag_class, tags):
        filtered = {}

        for tag in tags:

            # find stations that should be copied (match the filteR)
            keep = []
            for stat in self.data[G.STAT_NAMES]:
                this_tag = self.data[G.OBS][G.STAT][stat][G.STAT_META][tag_class].values
                if this_tag == tag:
                    keep.append(stat)

            # create sub data set and fill with stations that match the filter
            subdata = {}
            subdata[G.OBS] = {}
            subdata[G.MODEL] = {}
            subdata[G.STAT_NAMES] = keep
            for key in [G.OBS, G.MODEL]:
                subdata[key][G.DTS] = copy.deepcopy(self.data[key][G.DTS])
                subdata[key][G.PAR_NAMES] = copy.deepcopy(self.data[key][G.PAR_NAMES])
                subdata[key][G.STAT] = {}
                for stat in keep:
                    subdata[key][G.STAT][stat] = copy.deepcopy(self.data[key][G.STAT][stat])

            # save in filtered dictionary
            filtered[tag] = subdata

        return(filtered)

             
            

