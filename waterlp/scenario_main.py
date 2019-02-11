from datetime import datetime

from pyomo.opt import SolverStatus, TerminationCondition

from waterlp.reporters.post_reporter import Reporter as PostReporter
from waterlp.reporters.ably_reporter import AblyReporter
from waterlp.reporters.screen_reporter import ScreenReporter

def run_scenario(supersubscenario, args, verbose=False, **kwargs):

    system = supersubscenario.get('system')

    # setup the reporter (ably is on a per-process basis)
    post_reporter = PostReporter(args) if args.post_url else None
    reporter = None
    if args.message_protocol is None:
        reporter = ScreenReporter(args)
    elif args.message_protocol == 'post':
        post_reporter.is_main_reporter = True
        reporter = post_reporter
    elif args.message_protocol == 'ably':  # i.e. www.ably.io
        ably_auth_url = args.ably_auth_url if 'ably_auth_url' in args else kwargs.pop('ably_auth_url', None)
        reporter = AblyReporter(args, ably_auth_url=ably_auth_url, post_reporter=post_reporter)

    if reporter:
        reporter.updater = system.scenario.update_payload
        system.scenario.reporter = reporter

    if post_reporter:
        post_reporter.updater = system.scenario.update_payload

    try:

        # for result in _run_scenario(system, args, conn, supersubscenario, reporter=reporter, verbose=verbose):
        #     pass
        _run_scenario(system, args, supersubscenario, reporter=reporter, verbose=verbose)

    except Exception as err:

        print(err)

        if reporter:
            reporter.report(action='error', message=str(err))


def _run_scenario(system=None, args=None, supersubscenario=None, reporter=None, verbose=False):

    debug = args.debug

    # initialize with scenario
    # current_dates = system.dates[0:foresight_periods]

    # intialize
    system.initialize(supersubscenario)

    system.create_model(
        name=system.name,
        nodes=list(system.nodes.keys()),
        links=list(system.links.keys()),
        types=system.ttypes,
        ts_idx=system.ts_idx,
        params=system.params,
        blocks=system.blocks,
        debug_gain=args.debug_gain,
        debug_loss=args.debug_loss
    )

    runs = range(system.nruns)
    n = len(runs)

    now = datetime.now()

    i = 0
    while i < n:

        ts = runs[i]
        current_step = i + 1

        if verbose:
            print('current step: %s' % current_step)

        # if user requested to stop
        # if reporter._is_canceled:
        # print('canceled')
        # break

        #######################
        # CORE SCENARIO ROUTINE
        #######################

        # system.update_internal_params()

        # solve the model
        system.run(current_step, i, ts)

        # 6. REPORT PROGRESS

        new_now = datetime.now()
        should_report_progress = ts == 0 or current_step == n or (new_now - now).seconds >= 2
        # system.dates[ts].month != system.dates[ts - 1].month and (new_now - now).seconds >= 1

        if system.scenario.reporter and should_report_progress:
            system.scenario.reporter.report(action='step')

            now = new_now

        # update the model instance
        if ts != runs[-1]:
            ts_next = runs[i + 1]
            try:
                system.update_initial_conditions()
                system.update_boundary_conditions(ts, ts + system.foresight_periods, 'pre-process')
                system.update_boundary_conditions(ts_next, ts_next + system.foresight_periods, 'main')
            except Exception as err:
                # we can still save results to-date
                # system.save_results()
                msg = 'ERROR: Something went wrong at step {timestep} of {total} ({date}):\n\n{err}'.format(
                    timestep=current_step,
                    total=system.total_steps,
                    date=system.current_dates[0].date(),
                    err=err
                )
                print(msg)
                if system.scenario.reporter:
                    system.scenario.reporter.report(action='error', message=msg)

                raise Exception(msg)
            system.instance.preprocess()

        else:
            system.save_results()
            reporter and reporter.report(action='done')

            if verbose:
                print('finished')

        i += 1

        # yield

    # POSTPROCESSING HERE (IF ANY)

    # reporter.done(current_step, total_steps
