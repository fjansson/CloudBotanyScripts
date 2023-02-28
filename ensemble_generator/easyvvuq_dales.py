import os
import subprocess
import argparse
import sys
import easyvvuq as uq
import chaospy as cp
import matplotlib.pyplot as plt
from easyvvuq.decoders.json import JSONDecoder
from easyvvuq.encoders.jinja_encoder import JinjaEncoder
import numpy
import numpy.random
import json

# Analyzing DALES with EasyVVUQ
# based on the EasyVVUQ gauss tutorial
# Fredrik Jansson, CWI & TU Delft , 2019-2022

# 0. Setup some variables describing app to be run

cwd = os.getcwd()
inputdir = cwd+'/input'
input_filename = 'namoptions.001'
#out_file = "results.csv"; use_csv_decoder=True
out_file = "results.json"; use_csv_decoder=False # uncomment to use JSON-format results file
                                                  # which supports vector-valued QoIs (in progress)
#machine = 'eagle_vecma'  # for FabSim3
#machine = 'supermuc_vecma'  # for FabSim3

postproc = "postproc.py" # post-processing script, run after DALES for each sample

# Parameter handling
parser = argparse.ArgumentParser(description="EasyVVUQ for DALES",
                                 fromfile_prefix_chars='@')
parser.add_argument("--prepare",  action="store_true", default=False,
                    help="Prepare run directories")
parser.add_argument("--runsampler",  action="store_true", default=False,
                    help="Run a sampler, adding sample points to the database")
parser.add_argument("--addcenter",  action="store_true", default=False,
                    help="Add central point of parameter space")
parser.add_argument("--addpoints",  action="store_true", default=False,
                    help="Semi-manually add points in parameter space")
parser.add_argument("--run",  action="store_true", default=False,
                    help="Run model, sequentially")
parser.add_argument("--parallel", type=int, default=0,
                    help="use parallel command to run model with N threads")
parser.add_argument("--fab", action="store_true", default=False,
                    help="use Fabsim to run model")
parser.add_argument("--fetch",  action="store_true", default=False,
                    help="Fetch fabsim results")
parser.add_argument("--analyze",  action="store_true", default=False,
                    help="Analyze results")
parser.add_argument("--sampler",  default="sc", choices=['sc', 'pce', 'random'],
                    help="UQ sampling method, sc is the default.")
parser.add_argument("--num_samples",  default="10", type=int,
                    help="number of samples for the random sampler.")
parser.add_argument("--order",  default="2", type=int,
                    help="Sampler order")
parser.add_argument("--model",  default="dales4", help="Model executable file")
parser.add_argument("--workdir", default="/tmp", help="Model working directory base")
parser.add_argument("--template", default="namoptions.template", help="Template for model input file")
parser.add_argument("--campaign", default="", help="Campaign state file name")
parser.add_argument("--replicas", default="1", type=int, help="Number of replicas")
parser.add_argument("--experiment", default="physics_z0", help="experiment setup - chooses set of parameters to vary")
parser.add_argument("--plot", default=None, type=str, help="File name for plot")

args = parser.parse_args()
template = os.path.abspath(args.template)
print ("workdir:",args.workdir)

# 2. Parameter space definition. List of parameters that can be varied.
# (which parameters are actually varied in a single experiment is defined
# further below, vary = ...)
params = {
    "Nc_0": {
        "type": "float",
        "min": 0.1e6,
        "max": 1000e6,
        "default": 70e6,
    },
    # "cf": {  # cf subgrid filter constant
    #     "type": "float",
    #     "min": 1.0,     # min, max are just guesses
    #     "max": 4.0,
    #     "default": 2.5,
    # },
    # "cn": {  # Subfilterscale parameter
    #     "type": "float",
    #     "min": 0.4,     # min, max are just guesses
    #     "max": 1.0,
    #     "default": 0.76,
    # },
    # "Rigc": {  # Critical Richardson number
    #     "type": "float",
    #     "min": 0.09,     # min, max are just guesses
    #     "max": 1.0,
    #     "default": 0.25,
    # },
    # "Prandtl": {  # Prandtl number, subgrid.
    #     "type": "float",
    #     "min": 0.1,     # min, max are just guesses
    #     "max": 1.0,
    #     "default": 1.0/3,
    # },
    "z0": {            # surface roughness  - note: if z0mav, z0hav are specified, they will override z0
        "type": "float",
        "min": 1e-5,
        "max": 2.0,
        "default": 1.6e-4,
    },
    "l_sb": { # flag for microphysics scheme: false - KK00 Khairoutdinov and Kogan, 2000
        "type": "float",                   #   true - SB   Seifert and Beheng, 2001, 2006, Default
        "min" : 0,
        "max" : 1,
        "default": 1
    },
    "seed":{
        "type": "float",   # random seed
        "min" : 1,
        "max" : 1000000,
        "default" : 44
    },
    "poissondigits": {   # precision of the iterative Poisson solver. tolerance=10**-poissondigits
        "type": "float", # only useful if the template contains a &solver section
        "min": 1,
        "max": 16,
        "default": 15,
    },
    "ps": { # surface pressure, Pa
        "type": "float",
        "min": 90000,
        "max": 110000,
        "default": 101540.00,
    },
    "thls": { # sea surface temperature, K
        "type": "float",
        "min": 270,
        "max": 320,
        "default": 298.5, # 298.5,
    },
    "iadv": {
        "type": "float",
        "min": 0,
        "max": 1,
        "default": 0,
    },
    "iadv_sv": {
        "type": "float",
        "min": 0,
        "max": 2,
        "default": 2,
    },
    "dudz": {  # wind shear for generating initial wind profile
        "type": "float",
        "min": -1,
        "max": 1,
        "default": 0.0022,  # m/s / m
    },
    "u0": {  # surface wind for generating initial wind profile
        "type": "float",
        "min": -20,
        "max": 20,
        "default": -10,  # m/s
    },
#    "thl_high": {  # parameter for generating initial thl profile
#        "type": "float",
#        "min": 250.0,
#        "max": 400.0,
#        "default": 317.0,
#    },
#    "thl_low":{ # near-surface thl (K) for the initial profile
#        "type": "float",
#        "min": 250,
#        "max": 370,
#        "default" : 293.5,
#    },
#    "qt_high_delta": {  # perturbation parameter for initial qt profile
#        "type": "float",
#        "min": -0.005,
#        "max": 0.005,
#        "default": 0,
#    },
    "w0": {  # Subsidence (aloft). Downward motion is positive.
        "type": "float",
        "min": -0.1,
        "max": 0.1,
        "default": 4e-3,  # m/s
    },
    "wpamp": {  # Subsidence - amplitude of sine shape.
        "type": "float",
        "min": -0.1,
        "max": 0.1,
        "default": 0,  # m/s
    },
    "case": {  # Base profile selection. 0 - RICO, 1 - EUREC4A, 2 - exponential qt profile
        "type": "float",
        "min": 0,
        "max": 2,
        "default" : 0,
    },
    "iradiation": {         # Radiation scheme
        "type": "float",    #  0: prescribed cooling
        "min": 0,           #  1: RRTMG
        "max": 1,
        "default" : 0,
    },
    "qt0":{                  # just-above-surface qt (kg/kg) for exponential qt profile
        "type": "float",
        "min": 0,
        "max": 0.1,
        "default" : 0.016,
    },
    "qt_lambda":{        # qt decay length scale (m) for exponential qt profile
        "type": "float",
        "min": 1,
        "max": 10000,
        "default" : 1500,
    },
    "thl_Gamma":{        # lapse rate K/km of thl for case with linear thl profile
        "type": "float",
        "min": -10,
        "max":  20,
        "default" : 6,
    },
    "z_ml":{        # initial mixed-layer height (m), for the initial thl, qt profiles
        "type": "float",
        "min": 0,
        "max": 10000,
        "default" : 0,
    },
    "thl_tend0":{        #large-scale, advective thl tendency at surface. Linearly tapers off with height
        "type": "float", # K/s
        "min": -1e-4,
        "max": 1e-4,
        "default" : 0,   # -5.78e-6 K/s  = -0.5 K/day
    },
    "qt_tend0":{        #large-scale, advective qt tendency at surface. Linearly tapers off with height
        "type": "float",# kg/kg / s
        "min": -.1e-6,
        "max": .1e-6,
        "default" : 0,  # -1.73e-8 = -1.5 g/kg / day
    },
    "dthl0":{        #offset thls to lowest level thl
        "type": "float", # K
        "min": -10,
        "max":  10,
        "default" : 1.25,  # -1.73e-8 = -1.5 g/kg / day
    },
}

# The following is a list of experiment definitions, one of which can be
# selectend with the --experiment command line switch.

# vary physical quantities
# note: z0 has effect if z0hav, z0mav are not in namelist - use namoptions-z0.template
vary_physics_z0 = {
    "seed"    : cp.DiscreteUniform(1, 2000),
    "Nc_0"    : cp.Uniform(50e6, 100e6),
    "thls"    : cp.Uniform(298, 299),
    "z0"      : cp.Uniform(1e-4, 2e-4),
}

# vary subgrid scheme parameters
vary_subgrid = {
    "seed"    : cp.DiscreteUniform(1, 2000),
    "cn"      : cp.Uniform(0.5, 0.9),  # default 0.76
    "Rigc"    : cp.Uniform(0.1, 0.4),  # default 0.25
    "Prandtl" : cp.Uniform(0.2, 0.4),  # default 1/3
}

# vary microphysics choice, advection scheme
vary_choices = {
    "l_sb"    : cp.DiscreteUniform(0, 1),  # 0 - false, 1 - true
    "iadv"    : cp.DiscreteUniform(0, 1),  # 0 - 2nd order, 1 - 5th order
    "iadv_sv" : cp.DiscreteUniform(0, 2),  # 0 - 2nd order, 1 - 5th order, 2 - kappa scheme
    "seed"    : cp.DiscreteUniform(1, 2000),
}

# vary Poisson solver tolerance
# note use namoptions.poisson template which has iterative solver
vary_poisson = {
    "seed"    : cp.DiscreteUniform(1, 2000),
    "poissondigits": cp.Uniform(2,13),
    # the iteration doesn't always converge when poissondigits >= 14
}

# small test run
vary_test = {
    "Nc_0"    : cp.Uniform(50e6, 100e6),
    "seed"    : cp.DiscreteUniform(2, 2000),   # lower bound=1 and rule C triggers chaospy bug #304
}

vary_cloud_botany_7 = {
    "thls"          : cp.Uniform(297.5, 299.5),  # K  RICO has thls = 298.5K  EUREC4A has 299.135
    "u0"            : cp.Uniform(-15, -5),        # m/s
    "qt0"           : cp.Uniform(0.0135, 0.015),   # kg/kg
    "qt_lambda"     : cp.Uniform(1200, 2500),     # m
    "thl_Gamma"     : cp.Uniform(4.5, 5.5),       # K/km
#    "w0"            : cp.Uniform(0.0025, 0.0065),  # m/s  #
     "wpamp"       : cp.Uniform(-0.0035, 0.0018) # m/s
}

# map --experiment option to a dictionary of parameters to vary, and the polynomial order
# for each parameter. The number of samples along each parameter dimension is (order + 1)
# for the SC method.
experiment_options = {
    'physics_z0' : (vary_physics_z0, (3, 3, 3, 3)), # physics including z0
    'poisson'    : (vary_poisson, (4, 6)),
    'test'       : (vary_test,    (1, 1)),
    'choices'    : (vary_choices, (2,2,3,5)),    # advection and microphysics schemes
    'subgrid'    : (vary_subgrid, (2,2,2,2)),
    
    'cloud_botany_7': (vary_cloud_botany_7, (1,1,1,1,1,1), 'input-botany-6'),
}

vary, order = experiment_options[args.experiment][0:2]
if len(experiment_options[args.experiment]) > 2:
    inputdir = cwd+'/'+experiment_options[args.experiment][2]

if vary == vary_cloud_botany_7:
    params["case"]["default"] = 2
    params["iradiation"]["default"] = 1
    params["z_ml"]["default"] = 500 # m
#    params["thl_low"]["default"] = 294.5 # K
    params["dthl0"]["default"] = 1.25 # K offset  surf - first level
    params["w0"]["default"] = 0.0045 # m/s subsidence in free troposphere
#    params["z_w"]["default"]  = 2500 # m  length scale of subsidence exp
#    params["z_wp"]["default"] = 5300 # m  length scale of subsidence sin
    params["thl_tend0"]["default"] =  -5.78e-6  # K/s       = -0.5 K/day
    params["qt_tend0"]["default"] =   -1.73e-8  # kg/kg / s = -1.5 g/kg / day

print('Parameters chosen for variation:', vary)

# list of model output quantities of interest (QoIs) to analyze
output_columns = ['cfrac', 'lwp', 'rwp', 'zb', 'zi', 'prec', 'wq', 'wtheta', 'walltime', 'qt', 'ql', 'thl', 'zcfrac', 'u', 'v',
# the following are cloud metrics
'cf',
'cwp',
'lMax',
'periSum',
'cth',
'sizeExp',
'lMean',
'specLMom',
'cop',
'scai',
'nClouds',
'rdfMax',
'netVarDeg',
'iOrgPoiss',
'fracDim',
'iOrg',
'os',
'twpVar',
'cthVar',
'cwpVarCl',
'woi3',
'orie',
'd0',
'beta',
'betaa',
'specL',
'psdAzVar',
'cwpVar',
'cwpSke',
'cwpKur',
'eccA',
'cthSke',
'cthKur',
'rdfInt',
'rdfDiff',
'woi1',
'woi2',
'woi',
]
# omitted to save space: we

# dictionary of units of the different quantities
unit={
     'cfrac' :'',
     'lwp'   :'g/m$^2$',
     'rwp'   :'g/m$^2$',
     'zb'    :'km',
     'zi'    :'km',
     'prec'  :'W/m$^2$',
     'wq'    :'g/kg m/s',
     'wtheta':'K m/s',
     'we'    :'m/s',
     'z0'    :'mm',
     'Nc_0'  :'cm$^{-3}$',
     'walltime':'h',
     'ps'      :'Pa',
     'thls'    :'K',
     'ql'      :'g/kg',
     'qt'      :'g/kg',
     'thl'     :'K',
     'u'       :'m/s',
     'v'       :'m/s',
}

# unit conversion for some quantities for nicer display
scale={
    'lwp'      : 1000,     # convert kg/m^2 to g/m^2
    'rwp'      : 1000,     # convert kg/m^2 to g/m^2
    'wq'       : 1000,     # convert kg/kg m/s to g/kg m/s
    'zi'       : .001,     # convert m to km
    'zb'       : .001,     # convert m to km
    'Nc_0'     : 1e-6,     # convert m$^{-3}$, cm$^{-3}$,
    'walltime' : 1.0/3600, # convert seconds to hours
    'z0'       : 1000,     # convert m to mm
    'ql'       : 1000,     # convert kg/kg to g/kg
    'qt'       : 1000,     # convert kg/kg to g/kg
}

plot_labels = {
    'wtheta'   : r'$w_{\theta}$',
    'wq'       : '$w_q$',
    'we'       : '$w_e$',
    'walltime' : r'$\tau$',
    'zi'       : '$z_i$',
    'zb'       : '$z_b$',
    'iadv'     : 'adv.',
    'iadv_sv'  : 'rain adv.',
    'seed'     : 'seed',
    'l_sb'     : 'microphys.',
    'rwp'      : 'RWP',
    'lwp'      : 'LWP',
    'cfrac'    : '$C$',
    'prec'     : '$P_{srf}$',
    'z0'       : '$z_0$',
    'Nc_0'     : '$N_{c}$',
    'thls'     : r'$\theta_{s}$',
    'poissondigits' : '$d$',
    'cn'       : '$c_N$',      # Heus2010 cites Deardorf1980 for the subgrid formulation
    'Rigc'     : 'Ri$_c$',     # http://glossary.ametsoc.org/wiki/Bulk_richardson_number
    'Prandtl'  : 'Pr',
}


# adjust the order of discrete parameters
# to avoid repeating integer parameters with small range
#order = [args.order] * len(vary)
#for i,k in enumerate(vary):
#    #print(i, k, params[k])
#    if (params[k]["type"] == "integer"):
#        max_order = (params[k]["max"] - params[k]["min"])
#        order[i] = min(order[i],max_order)


print(f'Orders: {order} (only for SC sampler)')

# 4. Specify Sampler
if args.sampler=='sc':
    # sc sampler can have differet orders for different dimensions
    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=order,
                                       quadrature_rule="C",
                                       #quadrature_rule="grid",
                                       # sparse=True, growth=True
                                       )
elif args.sampler=='pce':
    print('order argument',args.order)
    my_sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=args.order)
                                        # quadrature_rule="G")
elif args.sampler=='random':
    my_sampler = uq.sampling.RandomSampler(vary=vary)
else:
    print("Unknown sampler specified", args.sampler)
    sys.exit()

def export_run_params(my_campaign):
    runs = {}
    # print the parameters of the runs, almost-JSON format with one row per run
    with open(os.path.join(my_campaign.campaign_dir, args.experiment + '-params.txt'), 'w') as pfile:
        for r in my_campaign.list_runs():
            runs[r[1]['run_name']] = r[1]['params']
            #print(r[1]['run_name'], r[1]['params'])
            print(r[1]['run_name'], r[1]['params'], file=pfile)

    # print the same as proper JSON
    # with indent=... each key gets its own row, nice but long.
    # print(json.dumps(runs, indent=4))
    with open(os.path.join(my_campaign.campaign_dir, args.experiment + '-params.json'), 'w') as pfile:
        json.dump(runs, pfile, indent=2)




    # 1. Create campaign - or load existing campaign
if args.campaign:
    my_campaign = uq.Campaign(name='botany-7-', work_dir=args.workdir,
                              db_location=args.campaign)
    resume = True
    print(f'Resuming {args.campaign} in {args.workdir}')
else:
    my_campaign = uq.Campaign(name='botany-7-',  work_dir=args.workdir)
    resume = False
    print('Setting up new campaign')

if args.prepare:

    # all run directories, and the database are created under workdir

    # 3. Wrap Application
    #    - Define a new application, and the encoding/decoding elements it needs
    #    - Also requires a collation element - this will be responsible for aggregating the results

    if not resume:
        # The encoder creates the input file to the model, from a template
        # The decoder reads model output into a database over runs
        # These are set only once when starting a campaign from scratch
        # Changing them, e.g. adding QoIs to collect, can be done (see recollate below)
        # but is not well supported.
        encoder = JinjaEncoder(template_fname=template,
                               target_filename=input_filename)
        if use_csv_decoder:
            decoder = uq.decoders.SimpleCSV(
                target_filename=out_file,
                output_columns=output_columns)
        else:
            decoder = JSONDecoder(
                target_filename=out_file,
                output_columns=output_columns)


        #        cmd = f"{work_dir}/gauss.py {input_filename}"
        #        execute = uq.actions.ExecuteLocal(
        #            "python3 {}".format(cmd)
        #        )


        # run pre-processing script for each run directory
        # used to copy input files that are common to each run
        # prep_script = 'prep.sh'
#        prep_script = f'{cwd}/prep-fugaku.sh'
#        prep_script_cmd = f"{prep_script} {inputdir} {cwd}"
        prep_script_cmd = f"{cwd}/case-setup-botany-7.py namoptions.001 "

        actions = uq.actions.Actions(
            uq.actions.CreateRunDirectory(root=args.workdir, flatten=True),
            uq.actions.Encode(encoder),
            uq.actions.ExecuteLocal(prep_script_cmd),
#            execute,
#            uq.actions.Decode(decoder)
        )

        my_campaign.add_app(name="dales",
                            params=params,
                            actions=actions,
#                            encoder=encoder,
#                            decoder=decoder,
        )

    my_campaign.set_sampler(my_sampler)

    if args.experiment=='choices':
        my_campaign.verify_all_runs = False
        # work-around to prevent validation errors on integer quantities
        # needed when *all* quantities varied are discrete

def add_point(params):
    # add one point in parameter space, if it's not already present

    # extend params dictionary with default values

    #my_campaign.set_app('dales')

    app_default_params = my_campaign._active_app["params"]
    extended_params = app_default_params.process_run(params,
                                                     verify=my_campaign.verify_all_runs)
    # note process_run seems to modify the input dict too (seems OK for us)

    # make a list of existing point parameters, to avoid adding duplicates
    # note this is sensitive to rounding
    all_runs = [ r[1]["params"] for r in my_campaign.list_runs() ]
    # print(all_runs)
    if extended_params not in all_runs:
        print("Adding: ", params)
        my_campaign.add_runs([params])
    else:
        print("Already present: ", params)

if args.addcenter:
    p_center = {
        "thls"        : 298.5,      # K
        "u0"          : -10,        # m/s
        "qt0"         : 0.01425,    # kg/kg
        "qt_lambda"   : 1850,       # m
        "thl_Gamma"   : 5.0,        # K/km
        "wpamp"       : -0.00085,    # m/s
    }
    p = p_center.copy()
    add_point(p)

if args.runsampler:
    # 5. Get run parameters
    if args.sampler=='random':
        my_campaign.draw_samples(num_samples=args.num_samples, replicas=args.replicas)
    else:
        #my_campaign.draw_samples(replicas=args.replicas)
        my_campaign.draw_samples()

    # 6. Create run input directories
    #my_campaign.populate_runs_dir()
    my_campaign.execute().collate()

    print(my_campaign)

    # list the (planned) runs and their parameters
    # wish: print only the varying ones in a nice table
    export_run_params(my_campaign)


################################################

if args.addpoints:

    p_center = {
        "thls"        : 298.5,      # K
        "u0"          : -10,        # m/s
        "qt0"         : 0.01425,    # kg/kg
        "qt_lambda"   : 1850,       # m
        "thl_Gamma"   : 5.0,        # K/km
        "wpamp"      : -0.00085,    # m/s
    }
    p = p_center.copy()
    for u in [-4.0, -5.0, -6.0, -8.0, -10.0, -12.0, -15.0]:
        p["u0"] = u
        add_point(p)
    print()
    p = p_center.copy()
    for w in [-0.002, -0.001, 0.0, 0.001]:
        p["wpamp"] = w
        add_point(p)
    print()
    p = p_center.copy()  # sweep thl_gamma
    for g in [4.0, 4.5, 4.75, 5.0, 5.25, 5.5, 6.0, 6.5, 7.5]:
        p["thl_Gamma"] = g
        add_point(p)
    print()
    p = p_center.copy()  # sweep shear
    for s in [-0.0044, -0.0033, -0.0022, -0.0011, 0, 0.0011, 0.0022, 0.0033, 0.0044]:
        p["dudz"] = s    # central point: 0.0022,  # m/s / m
        add_point(p)
    print()
    p = p_center.copy()  # sweep SST, central point 298.5K, range 297.5...299.5
    for t in [297.5, 298.5, 299.5, 300.5, 301.5]:
        p["thls"] = t
        add_point(p)
    print()
    p = p_center.copy()  # sweep qt_lambda, central point 1850m, range 1200...2500
    for l in [800, 1200, 1500, 1850, 2200, 2500, 3000]:
        p["qt_lambda"] = l
        add_point(p)
    print()
    p = p_center.copy()  # sweep qt0,  central point 0.01425,   range 0.0135, 0.015 # kg/kg
    for q in [0.0135, 0.01425, 0.015]:
        p["qt0"] = q
        add_point(p)

    my_campaign.execute().collate()

    export_run_params(my_campaign)

if args.run:
    # 7. Run Application
    #    - dales is executed for each sample


    if args.parallel:
        # run with gnu parallel, in parallel on the local machine
        pcmd = f"ls -d {my_campaign.campaign_dir}/runs/Run_* | parallel -j {args.parallel} 'cd {{}} ; {args.model} namoptions.001 > output.txt ;  cd .. '"
        print ('Parallel run command', pcmd)
        subprocess.call(pcmd, shell=True)
    elif args.fab: # run with FabSim
        # import fabsim3_cmd_api as fab
        fab.run_uq_ensemble(my_campaign.campaign_dir, script_name='dales', machine=machine)
    else:
        # run sequentially
        my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(f"/usr/bin/mpiexec -n 1 {args.model} namoptions.001 > output.txt"))



if args.fetch:
    if args.fab:
        print("Fetching results with FabSim:")
        fab.get_uq_samples(my_campaign.campaign_dir, machine=machine)

if args.analyze:

    # hack to collate old runs again after recollate
    #my_campaign.set_app("dales")
    #my_campaign.collate()
    #my_campaign.set_app("dales-sst300")
    #my_campaign.collate()

    # run post-processing script in each run directory
    # this is part of the job script on Cartesius
    # note this applies only to uncollated runs by default (despite the name)
    # https://easyvvuq.readthedocs.io/en/dev/easyvvuq.html#easyvvuq.campaign.Campaign.apply_for_each_run_dir
    # my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(postproc, interpret='python'))

    # 8. Collate output
    #my_campaign.collate()


    # to re-run all collation, use my_campaign.recollate()

    # try to create a new decoder, in case the output columns have changed
    # if use_csv_decoder:
    #    decoder = uq.decoders.SimpleCSV(
    #        target_filename=out_file,
    #        output_columns=output_columns)
    #else:
    #    decoder = JSONDecoder(
    #        target_filename=out_file,
    #        output_columns=output_columns)
    # my_campaign._active_app_decoder = decoder
    my_campaign.recollate()
    #my_campaign.save_state(args.campaign)


    data = my_campaign.get_collation_result()

    # try to get all past runs, not just the current set, for export
    # this lists all runs with parameters, but not results (yet)
    #run_list = my_campaign.list_runs()
    #print(run_list)

    # output_table_columns = ['qt_high_delta', 'wind_high', 'wind_low', 'thl_high', 'Nc_0', 'cf', 'cn', 'Rigc', 'Prandtl', 'z0', 'l_sb', 'seed', 'poissondigits', 'ps', 'thls', 'iadv', 'iadv_sv', 'cfrac', 'lwp', 'rwp', 'zb', 'zi', 'prec', 'wq', 'wtheta', 'walltime']
    output_table_columns = ['run_id', 'qt_high_delta', 'wind_high', 'wind_low', 'thl_high',
                            'seed', 'thls', 'w0', 'case', 'iradiation',
                            'cfrac', 'lwp', 'rwp', 'zb', 'zi', 'prec', 'wq', 'wtheta', 'walltime', 'iOrg']

    data.loc[:,output_table_columns].to_csv('datapoints.txt', index=None, sep=' ') #, float_format='% 6f')
    #data.loc[:,:].to_csv('datapoints.txt', index=None, sep=' ', float_format='% 6f')
    data.to_hdf('datapoints.h5', 'table')
    #sys.exit(1)

    # 9. Run Analysis
    if args.sampler == 'random':
        analysis = uq.analysis.BasicStats(qoi_cols=output_columns)
        my_campaign.apply_analysis(analysis)
        print("stats:\n", my_campaign.get_last_analysis())
        sys.exit()

    if args.sampler == 'sc':
        analysis = uq.analysis.SCAnalysis(sampler=my_sampler, qoi_cols=output_columns)
    elif args.sampler == 'pce':
        analysis = uq.analysis.PCEAnalysis(sampler=my_sampler, qoi_cols=output_columns)

    # perform analysis with EasyVVUQ
    my_campaign.apply_analysis(analysis)
    results = my_campaign.get_last_analysis()

    # from here on, it's reporting and plotting

    var = list(vary.keys()) # names of the parameters we vary
    if 'seed' in var:
        # put 'seed' last for consistency
        # cannot change the vary dict after the runs are already done (EasyVVUQ issue?)
        var.remove('seed')
        var.append('seed')

    print(f"sampler: {args.sampler}, order: {args.order}")
    print('         --- Varied input parameters ---')
    print("  param    default      unit     distribution")
    for k in var:
        print("%8s %9.3g %9s  %s"%(k, params[k]['default'], unit.get(k, ''), str(vary[k])))
    print()

    print('         --- Output ---')


    latex = True # output tables in LaTeX format
    if latex:
        sep=' &'       # column separator
        end=r' \\'     # end of line
        percent=r'\%'  # percent sign
    else:
        sep = ' '
        end=''
        percent='%'

    if latex:
        print('\\begin{tabular}{lrr*{%d}{r}}'%len(var))
        print(r'\hline')

    #print("                                                  Sobol indices")
    print(f"       QoI  {sep}    mean {sep}std({percent})", end='')
    for v in var:
        if latex:
            v = plot_labels.get(v, v)
        print(sep, '%12s'%v, end='')
    print(end)
    if latex:
        print(r'\hline')

    #print(results.describe())
    desc = results.describe()
    for qoi in output_columns:
        if latex:
            q = plot_labels.get(qoi, qoi)
        else:
            q = qoi

        print("%12s"%q, end=sep)
        m = desc[qoi]['mean'][0]
        s = desc[qoi]['std'][0]
        print("% 6.3g %9s%s% 6.1f"%(m * scale.get(qoi,1), unit.get(qoi,''), sep,
                       #% 6.3g%s             # results['statistical_moments'][qoi]['std'] * scale.get(qoi,1), sep, # st.dev.
                                               100*abs(s/m)),
              end='')
        #print("%9s"%unit[qoi], end='')
        for v in var:
            print('%s %5.3f'%(sep, results.sobols_first()[qoi][v]), end='')
        print(end)

    if latex:
        print(r'\hline')
        print(r'\end{tabular}')
    print()

    ## breakpoint
    # import pdb; pdb.set_trace()

    # print multi-variable Sobol indices - currently broken, interface is changing?
#    if args.sampler == 'sc':  # multi-var Sobol indices are not available for PCE
#        for qoi in output_columns:
#            print(qoi, end=' ')
#                  #results['statistical_moments'][qoi]['mean'][0],
#                  #results['statistical_moments'][qoi]['std'][0], end=' ')
#            #get_sobol_indices(qoi, 'all') # results['sobols'][qoi]
#            # sobols = results.sobols_second()
#            sobols = results.raw_data['sobols'][qoi]
#            for k in sobols:
#                if len(k) > 1: # print only the combined indices
#                    print(f"{k}: {sobols[k][0]:5.3f}", end=' ')
#            print()


    # print(my_campaign.get_collation_result()) # a Pandas dataframe

    mplparams = {#"figure.figsize" : [5.31, 5],  # figure size in inches
                 "figure.figsize" : [5.31, 20],  # figure size in inches
                 "figure.dpi"     :  200,      # figure dots per inch
                 "font.size"      :  6,        # this one acutally changes tick labels
                 'svg.fonttype'   : 'none',   # plot text as text - not paths or clones or other nonsense
                 'axes.linewidth' : .5,
                 'xtick.major.width' : .5,
                 'ytick.major.width' : .5,
                 'font.family' : 'sans-serif',
                 'font.sans-serif' : ['PT Sans'],
                 # mathmode font not yet set !
    }
    plt.rcParams.update(mplparams)

    scalar_outputs = ['cfrac', 'lwp', 'rwp', 'zb', # 'zi', 'prec', 'wq', 'wtheta', 'walltime', #'qt', 'ql', 'thl', 'zcfrac', 'u', 'v',
# the following are cloud metrics
#'cf',
#'cwp',
'lMax',
'periSum',
'cth',
#'sizeExp',
'lMean',
'specLMom',
#'cop',
'scai',
'nClouds',
'rdfMax',
#'netVarDeg',
'iOrgPoiss',
'fracDim',
'iOrg',
'os',
'twpVar',
'cthVar',
'cwpVarCl',
'woi3',
#'orie',
#'d0',
#'beta',
#'betaa',
#'specL',
#'psdAzVar',
'cwpVar',
'cwpSke',
'cwpKur',
'eccA',
'cthSke',
'cthKur',
'rdfInt',
'rdfDiff',
'woi1',
'woi2',
'woi',
]

#    scalar_outputs = output_columns # [:-1]
    params = var
    fig, ax = plt.subplots(nrows=len(scalar_outputs), ncols=len(params),
                           sharex='col', sharey='row', squeeze=False) # constrained_layout=True - sounds nice but didn't work
    # fig.set_tight_layout(True) - didn't work either.
    # layout adjustment at the end with subplots_adjust

    # manually specify tick locations for some parameters
    ticks = {
        'poissondigits' : [2,4,6,8,10,12],
        'iadv' : [0,1],
        'iadv_sv' : [0,1,2],
        'l_sb' : [0,1],
        'seed' : [],
    }
    # manually specify labels for some parameters
    ticklabels = {
        'iadv' : ['2nd', '5th'],
        'iadv_sv' : ['2nd', '5th', 'kappa'],
        'l_sb' : ['KK00', 'SB']
    }

    symbolsize = 1
    if len(params) == 2:
        symbolsize = 1.5 # larger symbols for the plot with fewer params

    # create grid of plots
    for i,param in enumerate(params):            # column
        for j,qoi in enumerate(scalar_outputs):  # row
            x = numpy.array(data[param]) * scale.get(param,1)
            y = numpy.array(data[qoi])   * scale.get(qoi,1)
            xr = max(x) - min(x)

            # add spread in x, to show point cloud better
            x += (numpy.random.rand(*x.shape) - .5) * xr * .05
            ax[j][i].plot(x, y, 'o', ms=symbolsize, mec='none', color='#ff8000')

            if param in ticks:
                ax[j][i].set_xticks(ticks[param])
                if param in ticklabels:
                    ax[j][i].set_xticklabels(ticklabels[param])

            # hide internal tick marks
            if i==0:
                ax[j][i].yaxis.set_ticks_position('left')
            else:
                ax[j][i].yaxis.set_ticks_position('none')
            if j==len(scalar_outputs)-1:
                ax[j][i].xaxis.set_ticks_position('bottom')
            else:
                ax[j][i].xaxis.set_ticks_position('none')


    # adding labels after all plots, hoping for better placement
    for i,param in enumerate(params):
        for j,qoi in enumerate(scalar_outputs):
            xu = unit.get(param,'')
            yu = unit.get(qoi,'')
            if xu: xu = f"({xu})"
            if yu: yu = f"({yu})"
            param_label = plot_labels.get(param, param)
            qoi_label = plot_labels.get(qoi, qoi)

            ax[j][i].set(xlabel=f"{param_label} {xu}")
            ax[j][i].set_ylabel(f"{qoi_label}", rotation=0)  #for y unit: \n{yu}
            ax[j][i].spines['top'].set_visible(False)
            ax[j][i].spines['right'].set_visible(False)
            ax[j][i].patch.set_visible(False) # remove background rectangle?

    for a in ax.flat:
        a.label_outer()
        a.ticklabel_format(axis='y', style='sci', scilimits=(-5,5), useOffset=None, useLocale=None, useMathText=True)

    plt.subplots_adjust(left=.1, top=.99, bottom=.1, right=.99, wspace=0, hspace=0)
    fig.patch.set_visible(False) # remove background rectangle?

    if args.plot:
        print('Saving plot as', args.plot)
        plt.savefig(args.plot)
    #plt.show()
