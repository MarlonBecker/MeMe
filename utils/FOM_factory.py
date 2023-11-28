import numpy as np


class FigureOfMeritFactory:
    """
        Central class to return all FOMs.
    """
    @staticmethod
    def simpleFOM(overlaps, powers):
        return overlaps[0][0]

    @staticmethod
    def simpleFOM_suppress_1(overlaps, powers):
        return overlaps[0][0] - overlaps[0][1]

    @staticmethod
    def powersplitter_50_50_surpress_wl_1(overlaps, powers):
        return overlaps[0][0] - overlaps[1][1]

    @staticmethod
    def powersplitter_50_50_harmonic_FOM(overlaps, powers):
        channel_1_fom = 1 - np.abs(0.5 - powers[0][0])
        channel_2_fom = 1 - np.abs(0.5 - powers[0][1])
        harmonic_mean = 2*channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)

        return (harmonic_mean*2-1)

    @staticmethod
    def powersplitter_75_25_harmonic_FOM(overlaps, powers):
        channel_1_fom = 1 - np.abs(0.75-powers[0][0])
        channel_2_fom = 1 - np.abs(0.25-powers[0][1])
        harmonic_mean = 2*channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)

        return (harmonic_mean*2-1)

    @staticmethod
    def powersplitter_90_10_harmonic_FOM(overlaps, powers):
        channel_1_fom = 1 - np.abs(0.9-powers[0][0])
        channel_2_fom = 1 - np.abs(0.1-powers[0][1])
        harmonic_mean = 2*channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)
        return (harmonic_mean*2-1)

    @staticmethod
    def powersplitter_90_10_exp_FOM(overlaps, powers):
        rel_dev_1 = np.abs(0.9 - powers[0][0]) / powers[0][0]
        rel_dev_2 = np.abs(0.1 - powers[0][1]) / powers[0][1]

        mean = np.mean([rel_dev_1,rel_dev_2])

        return 1/mean

    @staticmethod
    def powersplitter_90_10_experimental_limted(overlaps, powers):
        target_p0 = 0.9
        target_p1 = 0.1
        eps = 0.005
        slope_factor = 1/40

        geo_mean = (max(eps, abs(target_p0 - powers[0][0])) *  max(eps, abs(target_p1 - powers[0][1])))**0.5

        out = 1 - np.exp(-slope_factor * (powers[0][0]+powers[0][1]) / geo_mean)
        best_possible_out = 1 - np.exp(-slope_factor / eps)

        return out/best_possible_out

    @staticmethod
    def powersplitter_90_10_relativeEuclidean(overlaps, powers):
        # print("powers", powers)
        target_p0 = 0.9
        target_p1 = 0.1
        
        # d(p, p') = (p-p')/(1-p) if p > p' else (p'-p)/p
        d0 = abs(target_p0-powers[0][0])/(powers[0][0] + (powers[0][0] > target_p0)*(1 - 2* powers[0][0]))
        d1 = abs(target_p1-powers[0][1])/(powers[0][1] + (powers[0][1] > target_p1)*(1 - 2* powers[0][1]))
        
        return np.exp(-(d0+d1))

    @staticmethod
    def powersplitter_90_10_ratio(overlaps, powers):
        target_p0 = 0.9
        target_p1 = 0.1
        
        target_r = target_p0/target_p1
        r = powers[0][0]/powers[0][1]
        
        return np.exp(-((target_r/r + r/target_r)/2-1))*(powers[0][0]+powers[0][1])*((powers[0][1]+powers[0][0]) <= 1)

    @staticmethod
    def powersplitter_90_10_ratio_overlaps(overlaps, powers):
        target_p0 = 0.9
        target_p1 = 0.1
        
        target_r = target_p0/target_p1
        r = overlaps[0][0]/overlaps[0][1]
        
        return np.exp(-((target_r/r + r/target_r)/2-1))*(overlaps[0][0]+overlaps[0][1])*((overlaps[0][1]+overlaps[0][0]) <= 1)
    

    @staticmethod
    def powersplitter_90_10_experimental(overlaps, powers):
        rel_dev_1 = np.abs(0.9 - powers[0][0])
        rel_dev_2 = np.abs(0.1 - powers[0][1])

        if rel_dev_1 < 0.01:
            rel_dev_1 = 0.01
        if rel_dev_2 < 0.01:
            rel_dev_2 = 0.01

        def geo_mean(iterable):
            a = np.array(iterable)
            return a.prod()**(1.0/len(a))

        mean = geo_mean([rel_dev_1, rel_dev_2])

        return 1/mean * (powers[0][0]+powers[0][1])

    @staticmethod
    def powersplitter_70_30_experimental(overlaps, powers):
        rel_dev_1 = np.abs(0.7 - powers[0][0])
        rel_dev_2 = np.abs(0.3 - powers[0][1])

        if rel_dev_1 < 0.01:
            rel_dev_1 = 0.01
        if rel_dev_2 < 0.01:
            rel_dev_2 = 0.01

        def geo_mean(iterable):
            a = np.array(iterable)
            return a.prod()**(1.0/len(a))

        mean = geo_mean([rel_dev_1, rel_dev_2])

        return 1/mean * (powers[0][0]+powers[0][1])

    @staticmethod
    def modeDemultiplexerFOM(overlaps, powers):
        return (overlaps[0][0] + overlaps[1][1] - overlaps[0][1] - overlaps[1][0])/2

    @staticmethod
    def modeDemultiplexerHarmonicFOM(overlaps, powers):
        channel_1_fom = overlaps[0][0] - overlaps[1][0] + 1.0
        channel_2_fom = overlaps[1][1] - overlaps[0][1] + 1.0
        harmonic_mean = channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)
        normalized_harmonic_mean = 2.0*(harmonic_mean - 0.5)

        return normalized_harmonic_mean

    @staticmethod
    def mode_demultiplexer_harmonic_pwr(overlaps, powers):
        channel_1_fom = powers[0][0] - powers[1][0] + 1.0  # mode or channel fom???
        channel_2_fom = powers[1][1] - powers[0][1] + 1.0  # mode or channel fom???
        harmonic_mean = channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)
        normalized_harmonic_mean = 2.0*(harmonic_mean - 0.5)

        return normalized_harmonic_mean

    @staticmethod
    def wl_demultiplexer_harmonic_pwr(overlaps, powers):
        channel_1_fom = ((1-powers[0][0])/(1-powers[1][0]))  # mode or channel fom???
        channel_2_fom = ((1-powers[1][1])/(1-powers[0][1]))  # mode or channel fom???
        harmonic_mean = channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)
        normalized_harmonic_mean = 2.0*(harmonic_mean - 0.5)

        return normalized_harmonic_mean

    @staticmethod
    def wl_demultiplexer_harmonic_pwr_no_neg(overlaps, powers):
        channel_1_fom = ((1-powers[0][0])/(1-powers[1][0]))  # mode or channel fom???
        channel_2_fom = ((1-powers[1][1])/(1-powers[0][1]))  # mode or channel fom???
        harmonic_mean = channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)
        normalized_harmonic_mean = 2.0*(harmonic_mean - 0.5)

        if normalized_harmonic_mean < 0:
            return 0.0

        return normalized_harmonic_mean

    @staticmethod
    def filter_2WL_simple_FOM(overlaps, powers):
        return (overlaps[0][0] - overlaps[1][1])/2

    @staticmethod
    def filter_2WL_relative_FOM(overlaps, powers):
        eps = 0.00000000000001
        return (overlaps[0][0]/(overlaps[1][1]+eps))/10000

    @staticmethod
    def filter_2WL_relative_FOM_trans_exp_2_scale_10000(overlaps, powers):
        eps = 0.00000000000001
        return (overlaps[0][0]**2/(overlaps[1][1]+eps))/10000

    @staticmethod
    def filter_2WL_relative_FOM_trans_exp_2_scale_100(overlaps, powers):
        eps = 0.00000000000001
        return (overlaps[0][0]**2/(overlaps[1][1]+eps))/100

    @staticmethod
    def filter_2WL_relative_FOM_scale_100(overlaps, powers):
        eps = 0.00000000000001
        return (overlaps[0][0]/(overlaps[1][1]+eps))/100

    @staticmethod
    def filter_2WL_relative_FOM_exp32(overlaps, powers):
        eps = 0.00000000000001
        return overlaps[0][0]**32/((overlaps[1][1]+eps)**32)

    @staticmethod
    def filter_2WL_relative_FOM_inverse_difference(overlaps, powers):
        eps = 0.00000000000001
        powers = np.clip(powers, 0, 1)
        powers = np.around(powers, 3)
        return ((1-powers[0][0])/(1-powers[1][0]+eps))

    @staticmethod
    def attenuator_80(overlaps, powers):
        return (1 - np.abs(0.8 - powers[0][0]))**2

    @staticmethod
    def te00_te10_powersplitter(overlaps, powers):
        channel_1_fom = 1 - np.abs(0.5 - overlaps[0][0])
        channel_2_fom = 1 - np.abs(0.5 - overlaps[1][1])
        harmonic_mean = 2*channel_1_fom*channel_2_fom/(channel_1_fom+channel_2_fom)

        return (harmonic_mean*2-1)

    @staticmethod
    def power_focus_fom(overlaps, powers):
        return powers[0][0]

    @staticmethod
    def cross(overlaps, powers):
        return overlaps[0][0] + overlaps[1][1] - overlaps[0][1] - overlaps[1][0] + powers[0][0]

    @staticmethod
    def cross_dipole(overlaps, powers):
        return overlaps[0][0] + overlaps[1][1] - overlaps[0][1]*3 + powers[0][0]*5 + overlaps[2][1]

    @staticmethod
    def cross_qd_1(overlaps, powers):
        return (powers[0][0]) * overlaps[0][0] * overlaps[1][1] / powers[0][1]

    @staticmethod
    def cross_qd_2(overlaps, powers):
        return (powers[0][0]) * overlaps[0][0]**2 * overlaps[1][1] / powers[0][1]

    @staticmethod
    def cross_qd_3(overlaps, powers):
        return powers[0][0]**2 * overlaps[0][0] * overlaps[1][1]**2 / powers[0][1]

    @staticmethod
    def wl_demultiplexer_3_port_harmonic_overlaps(overlaps, powers):
        fom1 = (1 - overlaps[0][1] - overlaps[0][2])/(1 - overlaps[0][0])
        fom2 = (1 - overlaps[1][0] - overlaps[1][2])/(1 - overlaps[1][1])
        fom3 = (1 - overlaps[2][0] - overlaps[2][1])/(1 - overlaps[2][2])
        harmonic_mean = fom1*fom2*fom3/(fom1+fom2+fom3)
        normalized_harmonic_mean = 2.0*(harmonic_mean - 0.5)

        return normalized_harmonic_mean

    @staticmethod
    def mode_demultiplexer_3_port_harmonic_overlaps_normalized(overlaps, powers):
        fom1 = overlaps[0][0] - overlaps[0][1] - overlaps[0][2]
        fom2 = overlaps[1][1] - overlaps[1][0] - overlaps[1][2]
        fom3 = overlaps[2][2] - overlaps[2][1] - overlaps[2][0]

        fom1 = (fom1 + 1) * 0.5
        fom2 = (fom2 + 1) * 0.5
        fom3 = (fom3 + 1) * 0.5

        harmonic_mean = 3 / (1/fom1 + 1/fom2 + 1/fom3)

        return harmonic_mean
    
    @staticmethod
    def wl_demultiplexer_3_port_harmonic_overlaps_limited(overlaps, powers):
        fom1 = overlaps[0][0] - overlaps[0][1] - overlaps[0][2]
        fom2 = overlaps[1][1] - overlaps[1][0] - overlaps[1][2]
        fom3 = overlaps[2][2] - overlaps[2][1] - overlaps[2][0]

        fom1 = (fom1 + 1) * 0.5
        fom2 = (fom2 + 1) * 0.5
        fom3 = (fom3 + 1) * 0.5

        harmonic_mean = 3 / (1/fom1 + 1/fom2 + 1/fom3)

        return harmonic_mean