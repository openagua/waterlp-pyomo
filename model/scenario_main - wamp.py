import sys
import traceback
from io import StringIO
from os.path import join

from connection import connection
from model import create_model
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from post_reporter import Reporter
from systems import WaterSystem
from utils import create_logger

logd = None
current_step = 0
total_steps = 0

post_reporter = None  # global object to communicate with OpenAgua

statuses = {
    'start': 'started',
    'done': 'finished',
    'error': 'error',
    'pause': 'paused',
    'resume': 'resuming',
    'stop': 'stopped',
    'step': 'running',
    'save': 'saving'
}

def update_payload(payload={}, args=None, action=None):
    payload.update({'status': statuses.get(action)})
    if args:
        payload.update({
            'sid': args.unique_id,
            'source_id': args.source_id,
            'network_id': args.network_id,
            'scenario_ids': args.scenario_ids,
            'scenario_name': args.scenario_name        
        })
    if action:
        payload['action'] = action
    return payload

class Scenario(object):
    def __init__(self, scenario_ids, scenarios):
        self.base_scenarios = []
        self.source_scenarios = {}
    
        # look for existing option-scenario combination
        source_names = []
        self.tags = []
    
        # collect source IDs
        self.source_ids = []
        for i, base_id in enumerate(set(scenario_ids)):
            source = [s for s in scenarios if s.id == base_id][0]
            self.base_scenarios.append(source)
            self.source_scenarios[base_id] = source
    
            this_chain = [source.id]
    
            while source['layout'].get('parent'):
                parent_id = source['layout']['parent']
                if parent_id not in self.source_ids:  # prevent adding in Baseline twice, which would overwrite options
                    this_chain.append(parent_id)
                source = self.conn.call('get_scenario', {'scenario_id': parent_id})
                self.source_scenarios[source.id] = source
            this_chain.reverse()
            self.source_ids.extend(this_chain)
        
        self.base_ids = []
        for s in self.base_scenarios:
            self.base_ids.append(s.id)
            if s.layout.get('tags'):
                self.tags.extend(s.layout.tags)
                
            source_names.append(s.name)
    
        self.name = ' - '.join(source_names)
        if len(source_names)==1:
            self.name += ' (results)'
        #results_scenario_name = '{}; {}'.format(base_name, self.starttime.strftime('%Y-%m-%d %H:%M:%S'))
            
        self.option = self.base_scenarios[0]
        self.scenario = self.base_scenarios[-1]
        

def run_scenario(scenario_ids, args=None):
    global post_reporter, scenario_name, current_step, total_steps, logd, paused
    
    paused = False
    
    try:
        scenario_ids = list(scenario_ids)
    except:
        scenario_ids = [scenario_ids]    
    
    args.unique_id += '-' + '-'.join(str(s_id) for s_id in scenario_ids)
    
    logd = create_logger(appname='{} - {} - details'.format(args.app_name, scenario_ids),
                         logfile=join(args.scenario_log_dir, '{}.txt'.format(args.unique_id)),
                         msg_format='%(asctime)s - %(message)s')
    
    conn = connection(args=args, scenario_ids=scenario_ids, log=logd)
    scenario = Scenario(scenario_ids, conn.network.scenarios)
    args.scenario_name = scenario.name
    args.scenario_ids = scenario_ids
    
    post_reporter = Reporter(args)
    start_payload = update_payload(action='start', args=args)
    try:
        if getattr(args, 'websocket_url', None) is None:
            post_reporter.start(is_main_reporter=True, **start_payload) # kick off reporter with heartbeat
            for result in _run_scenario(args=args, conn=conn, scenario=scenario, report=post_reporter.report):
                pass
        else:
            from autobahn.asyncio.component import run, Component
            
            post_reporter.start(is_main_reporter=False, **start_payload)   # kick off reporter without heartbeat
            
            update_channel = u'com.openagua.update_s{}n{}'.format(args.source_id, args.network_id)
            
            component = Component(
                transports=u"{}".format(args.websocket_url),
                realm=u"realm1"
            )
            
            args.scenario_ids = scenario_ids
           
            @component.on_join
            def joined(session, details):
                
                # publish updates
                def report(action, **payload):
                    payload = update_payload(action=action, payload=payload, args=args)
                    session.publish(update_channel, payload)
                    if action != 'step':
                        post_reporter.report(**payload)
                    if action in ['done', 'error']:
                        session.leave()
                        return
                
                # subscribe to actions
                def on_action(msg):
                    action = msg['action']
                    if action == 'stopall' or msg['action'] == 'stop' and msg['sid'] == args.unique_id:
                        report(action='stop', progress=current_step / total_steps * 100)
                        session.disconnect()
                    elif action == 'pause':
                        paused = True
                        report(action='pause', progress=current_step / total_steps * 100)
                    elif action == 'resume':
                        paused = False
                        report(action='resume')
                                            
                session.subscribe(on_action, u'com.openagua.action_s{}n{}'.format(args.source_id, args.network_id))             
                        
                # run the model
                yield from _run_scenario(args=args, conn=conn, scenario=scenario, report=report, session=session)
           
            run([component])
            
    except Exception as e:

        #print(e, file=sys.stderr)
        # Exception logging inspired by: https://seasonofcode.com/posts/python-multiprocessing-and-exceptions.html
        exc_buffer = StringIO()
        traceback.print_exc(file=exc_buffer)
        err = 'At step ' + str(current_step) + ' of ' + str(total_steps) + ': ' + \
              str(e) + '\nUncaught exception in worker process:\n' + exc_buffer.getvalue()
        if current_step:
            err += '\n\nPartial results have been saved'
        payload = update_payload(payload={'message': err}, args=args, action='error')
        post_reporter.report(**payload)
        logd.error(err)

def _run_scenario(args=None, scenario=None, conn=None, report=None, session=None):
    global logd, current_step, total_steps
    
    debug = args.debug
    
    logd.info('starting new run, scenario: ' + str(scenario.name))

    ## START MODELING ROUTINE HERE

    # TIME STEP SETUP AND PREPROCESSING

    # get connection, along with useful tools attached
    #try:
        #scenario_ids = list(scenario_ids)
    #except:
        #scenario_ids = [scenario_ids]

    # create a dictionary of network attributes
    template_attributes = conn.call('get_template_attributes', {'template_id': conn.template.id})

    attrs = {ta.id: {'name': ta.name.replace(' ', '')} for ta in template_attributes}

    # START CORE MODEL ROUTINE

    # create the system class (this can be passed from main)
    system = WaterSystem(
        #starttime=args.initial_timestep,
        conn=conn,
        session=session,
        name=args.app_name,
        scenario=scenario,
        network=conn.network,
        template=conn.template,
        attrs=attrs,
        settings=conn.network.layout.get('settings'),
        #timestep_format=args.timestep_format,
        report=report,  # this is a verb (function), not a noun
        args=args,
    )
    
    system.logd = logd

    if debug:
        system.dates = system.dates[:5]

    # define foresight periods & timesteps
    # NB: to be as efficient as possible within run loops, we should keep as much out of the loops as possible
    nruns = len(system.dates)
    args.foresight = args.foresight if 'foresight' in args else 'zero' # pending further development
    if args.foresight == 'perfect':
        foresight_periods = len(system.dates)
        save_periods = foresight_periods
        nruns = 1
    elif args.foresight == 'zero':
        foresight_periods = 1
        save_periods = 1
    system.ts_idx=range(foresight_periods)

    if nruns == 1 or debug:
        verbose = False
    else:
        verbose = False

    # initialize with scenario
    current_dates = system.dates[0:foresight_periods]

    # gather
    system.collect_source_data(tsi=0, tsf=foresight_periods)
    
    # intialize
    system.update_variables(0, foresight_periods, initialize=True)
    system.prepare_params()
    system.model = create_model(
        name=system.name,
        template=system.template,
        nodes=list(system.nodes.keys()),
        links=list(system.links.keys()),
        types=system.types,
        ts_idx=system.ts_idx,
        params=system.params,
        blocks=system.blocks,
        debug=debug
    )
    
    system.instance = system.model.create_instance()

    system.update_internal_params()

    optimizer = SolverFactory(args.solver)
    logd.info('Model created.')
    logd.info('Starting model run.')

    total_steps = len(system.dates)

    failed = False

    runs = range(nruns)

    #last_year = arrow.get(system.timesteps[0]).year

    i = 0
    while i < len(runs):
    #for i, ts in enumerate(runs):
        
        if paused:
            time.sleep(0.5)  # avoid a tight loop
            continue            
        else:
            ts = runs[i]
            current_step = i + 1       

        # if user requested to stop
        #if reporter._is_canceled:
            #print('canceled')
            #break
            
        #######################
        # CORE SCENARIO ROUTINE
        #######################

        current_dates = system.dates[ts:ts+foresight_periods]
        current_dates_as_string = system.dates_as_string[ts:ts+foresight_periods]

        # solve the model
        results = optimizer.solve(system.instance)
        #system.instance.solutions.load_from(results)
            
        # print & save summary results
        if verbose:
            old_stdout = sys.stdout
            sys.stdout = summary = StringIO()
            logd.info('model solved\n' + summary.getvalue())

        if (results.solver.status == SolverStatus.ok) \
           and (results.solver.termination_condition == TerminationCondition.optimal):
            # this is feasible and optimal
            if verbose:
                logd.info('Optimal feasible solution found.')

            system.collect_results(current_dates_as_string, write_input=args.write_input)

            if verbose:
                logd.info('Results saved.')

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            system.save_results()
            msg = 'ERROR: Problem is infeasible at step {} of {} ({}). Prior results have been saved.'.format(
                current_step, total_steps, current_dates[0]
            )
            logd.info(msg)
            report(action='error', message=msg)
            break

        else:
            system.save_results()
            # something else is wrong
            msg = 'ERROR: Something went wrong. Likely the model was not built correctly.'
            print(msg)
            logd.info(msg)
            report(action='error', message=msg)
            break

        #if foresight_periods == 1:
        #print("Writing results...")
        #results.write()

        #else:

        # load the results
        #print("Loading results...")
        system.instance.solutions.load_from(results)
        if verbose:
            sys.stdout = old_stdout
            
        # report progress
        report(action='step', status='running', progress=current_step / total_steps * 100)
            
        # update the model instance
        if ts != runs[-1]:
            ts_next = runs[i+1]
            system.update_initial_conditions()
            system.update_variables(ts, ts+foresight_periods) # SLOW
            system.update_internal_params() # update internal parameters that depend on user-defined variables
            system.instance.preprocess()

        else:
            system.save_results()
            action = 'done'
            report(action=action, status=statuses[action], progress=current_step / total_steps * 100)
            logd.info('done run')

        if verbose:
            logd.info(
                'completed timestep {date} | {timestep}/{total_timesteps}'.format(
                    date=system.dates[ts],
                    timestep=ts+1,
                    total_timesteps=nruns
                )
            )

        #######################
        
        i += 1
        
        yield

    # POSTPROCESSING HERE (IF ANY)

    #reporter.done(current_step, total_steps)
