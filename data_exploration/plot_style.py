import matplotlib.pyplot as plt

def apply_academic_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 1.2,
        
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        
        "axes.grid": False,            
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,      
        
        "savefig.dpi": 300,            
        "savefig.bbox": "tight"      
    })
    
apply_academic_style()