import numpy as np
import copy
import globals as G

class StationFilter:

    def __init__(self):
        pass

    
    def _create_subdataset(self, keep, data):
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

        return(subdata)


    def filter_according_tag(self, data, tag_class, tags):
        filtered = {}

        for tag in tags:

            # find stations that should be copied (match the filteR)
            keep = []
            for stat in data[G.STAT_NAMES]:
                this_tag = data[G.OBS][G.STAT][stat][G.STAT_META][tag_class].values
                if this_tag == tag:
                    keep.append(stat)

            # create subdata set save in filtered dictionary
            filtered[tag] = self._create_subdataset(keep,data)

        return(filtered)


             
    def filter_according_altitude(self, data, altitudes):
        filtered = {}

        for i,alt_lims in enumerate(altitudes):
            print(alt_lims)

            # find stations that should be copied (match the filteR)
            keep = []
            for stat in data[G.STAT_NAMES]:
                this_alt = data[G.OBS][G.STAT][stat][G.STAT_META]['Height'].values
                if (this_alt >= alt_lims[0]) and (this_alt < alt_lims[1]):
                    keep.append(stat)

            # create subdata set save in filtered dictionary
            tag = str(alt_lims)
            filtered[tag] = self._create_subdataset(keep,data)

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

             

    #def filter_according_mean_wind_acc(self, data, rel_acc):

    #    for stat in data[G.STAT_NAMES]:

    #        wind_obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M'].values
    #        wind_mod = data[G.MODEL][G.STAT][stat][G.GUST]['VMAX_10M1'].values
    #        mask = gust_obs < min_gust

    #        # remove in obs
    #        obs_fields = data[G.OBS][G.PAR_NAMES]
    #        for field in obs_fields:
    #            data[G.OBS][G.STAT][stat][G.PAR][field][mask] = np.nan

    #        # remove in model
    #        mod_gust_fields = list(data[G.MODEL][G.STAT][stat][G.GUST].keys())
    #        for field in mod_gust_fields:
    #            data[G.MODEL][G.STAT][stat][G.GUST][field][mask] = np.nan

    #    return(data)
            

