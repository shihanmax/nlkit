from collections import Counter

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def explore_data(arr, show_dist=True):
    """Get statistics on `arr`.
    
    get length/min/max/avg/25/50/75quantile/skew/kurtosis of `arr`
    
    if arr is a list of string, the min/max/avg/25/50/75quantile will 
    be not available.

    Args:
        arr (List): could be list of [int, float] or str
        show_dist (bool, optional): Whether to show distribution. 
            Defaults to True.
    """
    if not arr:
        return
    
    from_string = False
    
    if isinstance(arr[0], str):
        mapper = {}
        mapped_arr = []
        cnt = 0
        for i in arr:
            if i not in mapper:
                mapper[i] = cnt
                cnt += 1
            
            mapped_arr.append(mapper.get(i))
    
        arr = mapped_arr
        from_string = True
    
    if isinstance(arr[0], (int, float)):
        arr = np.array(arr)
        
        info = {
            "total": len(arr),
            "max": arr.max(),
            "min": arr.min(),
            "mean": arr.mean(),
            "25%": np.quantile(arr, 0.25), 
            "50%": np.quantile(arr, 0.5), 
            "75%": np.quantile(arr, 0.75),
            "skew": stats.skew(arr),  # 偏度
            "kurtosis": stats.kurtosis(arr),  # 峰度
        }
        
        for k, v in info.items():
            info[k] = round(v, 4)
            
            if k not in {"skew", "kurtosis", "total"} and from_string:
                info[k] = "-"
        
        print("====== stat ======")
        for k, v in info.items():
            print(f" {k}: {v}")
        print("==================\n")

    if not show_dist:
        return
    
    sorted_counter = sorted(Counter(arr).items(), key=lambda x: x[1])
    x = [i[0] for i in sorted_counter]
    y = [i[1] for i in sorted_counter]

    plt.bar(x, y, color="orange")
    plt.title("Freq. count")
    plt.show()
