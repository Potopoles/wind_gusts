import numpy as np
import copy
import globals as G

class StationFilter:

    def __init__(self):
        pass


    def filter_according_tag(self, data, tag_class, tags):
        filtered = {}

        for tag in tags:

            # find stations that should be copied (match the filteR)
            keep = []
            for stat in data[G.STAT_NAMES]:
                this_tag = data[G.OBS][G.STAT][stat][G.STAT_META][tag_class].values
                if this_tag == tag:
                    keep.append(stat)

            # create sub data set and fill with stations that match the filter
            subdata = {}
            subdata[G.OBS] = {}
            subdata[G.MODEL] = {}
            subdata[G.STAT_NAMES] = keep
            for key in [G.OBS, G.MODEL]:
                subdata[key][G.DTS] = copy.deepcopy(data[key][G.DTS])
                subdata[key][G.PAR_NAMES] = copy.deepcopy(data[key][G.PAR_NAMES])
                subdata[key][G.STAT] = {}
                for stat in keep:
                    subdata[key][G.STAT][stat] = copy.deepcopy(data[key][G.STAT][stat])

            # save in filtered dictionary
            filtered[tag] = subdata

        return(filtered)

             
            

class EntryFilter:

    def __init__(self):
        pass


    def filter_according_obs_gust(self, data, min_gust, ):

        for stat in data[G.STAT_NAMES]:

            gust_obs = data[G.OBS][G.STAT][stat][G.PAR]['VMAX_10M1'].values
            mask = gust_obs < min_gust

            # remove in obs
            obs_fields = data[G.OBS][G.PAR_NAMES]
            for field in obs_fields:
                data[G.OBS][G.STAT][stat][G.PAR][field][mask] = np.nan

            # remove in model
            mod_gust_fields = list(data[G.MODEL][G.STAT][stat][G.GUST].keys())
            for field in mod_gust_fields:
                data[G.MODEL][G.STAT][stat][G.GUST][field][mask] = np.nan

        return(data)

             
            

