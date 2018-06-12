import numpy as np
import copy
import globals as G
from functions import check_prerequisites

class StationFilter:

    def __init__(self):
        pass

    
    def _create_subdataset(self, keep, data):
        # create sub data set and fill with stations that match the filter
        subdata = {}
        subdata[G.OBS] = {}
        subdata[G.MODEL] = {}
        subdata[G.STAT_NAMES] = keep
        subdata[G.HIST] = copy.deepcopy(data[G.HIST])

        subdata[G.OBS][G.STAT] = {}
        subdata[G.MODEL][G.STAT] = {}

        for stat in keep:
            subdata[G.STAT_META] = copy.deepcopy(data[G.STAT_META][stat])
            subdata[G.OBS][G.STAT][stat] = copy.deepcopy(data[G.OBS][G.STAT][stat])

            subdata[G.MODEL][G.STAT][stat] = {}
            subdata[G.MODEL][G.STAT][stat][G.RAW] = {}
            for lm_run in list(data[G.MODEL][G.STAT][stat][G.RAW].keys()):
                subdata[G.MODEL][G.STAT][stat][G.RAW][lm_run] = \
                        copy.deepcopy(data[G.MODEL][G.STAT][stat][G.RAW][lm_run])

        return(subdata)


    def filter_according_tag(self, data, tag_class, tags):
        """
        For data (directly after loading only!) creates a dictionary containing
        for each tag in tag_class an entry with key = tag and value = subset of
        data of stations that have this tag.

        INPUT
        data:           dictionary
        tag_class:      (str) type of tags that should be filtered after (SfcTag or TopoTag)
        tags:           (list) of tags that should be used.

        OUTPUT
        filtered:       dictionary containing for each tag a subset of data of stations of that tag.
        """
        hist_tag = 'filter_according_tag'
        prerequisites = ['01_prep_obs', '02_prep_model']
        data = check_prerequisites(data, prerequisites, hist_tag)

        filtered = {}
        for tag in tags:

            # find stations that should be copied (match the filteR)
            keep = []
            for stat in data[G.STAT_NAMES]:
                this_tag = data[G.STAT_META][stat][tag_class].values
                if this_tag == tag:
                    keep.append(stat)

            # create subdata set save in filtered dictionary
            filtered[tag] = self._create_subdataset(keep,data)

        return(filtered)


             
    def filter_according_altitude(self, data, altitudes):
        """
        For data (directly after loading only!) creates a dictionary containing
        for each altitude group in altitudes an entry with key = alt_lims and
        value = subset of data of stations that lie within alt_lims.

        INPUT
        data:           dictionary
        altitudes:      (lst) of altitude limits to filter after

        OUTPUT
        filtered:       dictionary containing for each tag a subset of data of stations of that altitudes.
        """
        hist_tag = 'filter_according_altitude'
        prerequisites = ['01_prep_obs', '02_prep_model']
        data = check_prerequisites(data, prerequisites, hist_tag)

        filtered = {}
        for i,alt_lims in enumerate(altitudes):
            print(alt_lims)

            # find stations that should be copied (match the filteR)
            keep = []
            for stat in data[G.STAT_NAMES]:
                this_alt = data[G.STAT_META][stat]['Height'].values
                if (this_alt >= alt_lims[0]) and (this_alt < alt_lims[1]):
                    keep.append(stat)

            # create subdata set save in filtered dictionary
            tag = str(alt_lims)
            filtered[tag] = self._create_subdataset(keep,data)

        return(filtered)
            








class EntryFilter:

    def __init__(self):
        pass


    def filter_according_obs_gust(self, data, min_gust):
        """
        For data[G.BOTH] filters value with observed gusts < min_gust

        INPUT
        data:           dictionary containing [G.BOTH]
        min_gust:       gust threshold value below which entries are removed

        OUTPUT
        data:           data with data[G.BOTH] altered according to filter
        """
        hist_tag = 'filter_according_obs_gust'
        prerequisites = ['01_prep_obs', '02_prep_model',
                        'calc_model_fields', 'join_model_and_obs']
        data = check_prerequisites(data, prerequisites, hist_tag)

        for stat in data[G.STAT_NAMES]:

            if 'join_model_runs' in data[G.HIST]:
                both = data[G.BOTH][G.STAT][stat]
                both = both[both[G.OBS_GUST_SPEED] >= min_gust]
                data[G.BOTH][G.STAT][stat] = both
            else:
                for lm_run in list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys()):
                    both = data[G.BOTH][G.STAT][stat][lm_run]
                    both = both[both[G.OBS_GUST_SPEED] >= min_gust]
                    data[G.BOTH][G.STAT][stat][lm_run] = both
            
        return(data)

             

    #def filter_according_mean_wind_acc(self, data, rel_acc):

    #    for stat in data[G.STAT_NAMES]:

    #        wind_obs = data[G.OBS][G.STAT][stat][G.PAR]['FF_10M'].values
    #        wind_mod = data[G.MODEL][G.STAT][stat][G.FIELDS][G.MEAN_WIND]
    #        abs_diff = np.abs(wind_obs - wind_mod)

    #        mask = abs_diff > rel_acc*wind_obs
    #        mask[np.isnan(wind_obs)] = True
    #        #print('remove ' + str(np.sum(mask)) + ' values du to inaccurate mean wind')

    #        # remove in obs
    #        obs_fields = data[G.OBS][G.PAR_NAMES]
    #        for field in obs_fields:
    #            data[G.OBS][G.STAT][stat][G.PAR][field][mask] = np.nan

    #        # remove in model
    #        mod_gust_fields = list(data[G.MODEL][G.STAT][stat][G.FIELDS].keys())
    #        for field in mod_gust_fields:
    #            data[G.MODEL][G.STAT][stat][G.FIELDS][field][mask] = np.nan

    #    return(data)
            

