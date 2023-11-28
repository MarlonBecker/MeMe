import numpy as np
from scipy.signal import convolve2d

def get_struct_regularizer(identifier: str):
    if identifier == "granularity":
        def regularizer(structure, returnChangeMatrix = False):
            kernel = np.ones((3,3))/8
            kernel[1,1] = -1
            fillvalue = -1
            gran_matrix = np.abs(convolve2d(structure*2-1, kernel, mode = "same", boundary = "fill", fillvalue=fillvalue))
            if returnChangeMatrix:
                missingBorderPixelMatrix = np.zeros(structure.shape)
                missingBorderPixelMatrix[0] = 3
                missingBorderPixelMatrix[-1] = 3
                missingBorderPixelMatrix[:, 0] = 3
                missingBorderPixelMatrix[:, -1] = 3
                missingBorderPixelMatrix[0,0] = 5
                missingBorderPixelMatrix[-1,0] = 5
                missingBorderPixelMatrix[0,-1] = 5
                missingBorderPixelMatrix[-1,-1] = 5
                return -4*(gran_matrix-1+fillvalue*missingBorderPixelMatrix/16*(structure*2-1))/structure.size #border handling makes this messy...
            return np.sum(gran_matrix)/structure.size

    elif identifier == "granularityThresholded":
        def regularizer(structure, returnChangeMatrix = False):
            if returnChangeMatrix:
                raise NotImplementedError("returnChangeMatrix=True not supported for structure_reg 'granularityThresholded' atm.")
            kernel = np.ones((3,3))/8
            kernel[1,1] = -1
            fillvalue = -1
            gran_matrix = np.abs(convolve2d(structure*2-1, kernel, mode = "same", boundary = "fill", fillvalue=fillvalue))
            neighbors_to_ignore = 4
            return np.sum(np.maximum(gran_matrix-neighbors_to_ignore/4, 0))/structure.size

    elif identifier == "granularityNormalized":
        def regularizer(structure, returnChangeMatrix = False):
            if returnChangeMatrix:
                raise NotImplementedError("returnChangeMatrix=True not supported for structure_reg 'granularityNormalized' atm.")
            kernel = np.ones((3,3))/8
            kernel[1,1] = -1
            fillvalue = -1
            gran_matrix = np.abs(convolve2d(structure*2-1, kernel, mode = "same", boundary = "fill", fillvalue=fillvalue))
            blanko_gran = ((structure.shape[0]-2) * 2 + (structure.shape[1]-2) * 2)*0.75+1.25*4
            pixels_ratio = np.sum(structure)/structure.size
            r = (np.sum(gran_matrix) - blanko_gran)/structure.size/min(pixels_ratio, 1-pixels_ratio)
            return r

    else:
        raise RuntimeError(f"Structure regularizer '{identifier}' not found.")
    return regularizer
