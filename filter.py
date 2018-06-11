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

            obs = data[G.OBS][G.STAT][stat]
            
            mask = obs[obs['VMAX_10M1'].values < min_gust].index
            obs = obs.drop(mask)

            for lm_run in list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys()):
                model = data[G.MODEL][G.STAT][stat][G.FIELDS][lm_run]
                this_mask = mask[np.in1d(mask,model.index)]
                model = model.drop(this_mask)
                data[G.MODEL][G.STAT][stat][G.FIELDS][lm_run] = model



            print(data[G.MODEL][G.STAT][stat][G.FIELDS])
            print(data[G.OBS][G.STAT][stat][G.FIELDS])
            quit()

        return(data)

             

    def filter_according_mean_wind_acc(self, data, rel_acc):

        for stat in data[G.STAT_NAMES]:

            wind_obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M'].values
            wind_mod = data[G.MODEL][G.STAT][stat][G.FIELDS][G.MEAN_WIND]
            abs_diff = np.abs(wind_obs - wind_mod)

            mask = abs_diff > rel_acc*wind_obs
            mask[np.isnan(wind_obs)] = True
            #print('remove ' + str(np.sum(mask)) + ' values du to inaccurate mean wind')

            # remove in obs
            obs_fields = data[G.OBS][G.PAR_NAMES]
            for field in obs_fields:
                data[G.OBS][G.STAT][stat][G.PAR][field][mask] = np.nan

            # remove in model
            mod_gust_fields = list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys())
            for field in mod_gust_fields:
                data[G.MODEL][G.STAT][stat][G.FIELDS][field][mask] = np.nan

        return(data)
            

