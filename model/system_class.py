import json
import sys
from collections import OrderedDict
import pandas as pd
from evaluator import Evaluator
from pyomo.environ import Var, Param
#from pymongo import MongoClient

def convert_type_name(n):
    n = n.title()
    for char in [' ', '/', '-']:
        n = n.replace(char, '')
    return n

def convert_attr_name(n):
    n = n.title()
    for char in [' ', '/', '-']:
        n = n.replace(char, '')
    return n

def get_param_name(resource_type, attr_name):
    # IMPORTANT! rt-name combinations should be unique! This can be resolved in two ways:
    # 1. append the dimension to the param name, or
    # 2. use a unique internal name for all variables (e.g., reservoir_demand, urban_demand)
    # then use this unique internal name instead of the rt-name scheme.
    # Number two is var preferable
    
    return '{rt}{name}'.format(rt=resource_type.lower(), name=convert_attr_name(attr_name))

def perturb(val, variation):
    # NB: this is made explicit to avoid using exec
    operator = variation['operator']
    value = variation['value']
    if operator == 'multiply':
        return val * value
    elif operator == 'add':
        return val + value
    else:
        return val

class Recorder(object):
    def __init__(self, flavor, host, port, database, username=None, password=None):
        
        self.flavor=flavor
        if flavor=='mongodb':
            client = MongoClient(host, port)
            db = client[database]
            self.results_collection = db['results']
            
    def record(self, records):
        
        if self.flavor=='mongodb':
            if type(records) != list:
                records = list(records)
            try:
                self.results_collection.insert_many(records)
            except:
                pass
            

class WaterSystem(object):

    def __init__(self, conn, name, network, all_scenarios, template, attrs, settings, args, timestep_format=None, session=None, reporter=None, scenario=None):
        
        self.VOLUMETRIC_FLOW_RATE_CONST = 0.0864  # 60*60*24/1e6
        
        self.conn = conn
        self.session = session
        self.name = name
        self.scenario = scenario
        self.template = template
        self.attrs = attrs
        self.reporter = reporter
        self.args = args

        self.scenarios = {s.name: s for s in all_scenarios.scenarios}

        self.evaluator = Evaluator(self.conn, settings=settings)
        self.dates = self.evaluator.dates
        self.dates_as_string = self.evaluator.dates_as_string
                        
        # TODO - don't really need this, only used for verbose logging
        #self.timesteps_formatted =  self.timesteps
                
        # SETUP THE RESULTS / "RECORDER"
        # This should be the originating database or some other database store
                
        flavor='mongodb'
        host = '127.17.0.3'
        port = 27017
        database = 'openagua'
        username=None
        password=None
        #self.recorder = Recorder(flavor=flavor, host=host, port=port, database=database, username=username, password=password)
        # TODO: we can pass the recorder from outside

        # timestep deltas
        self.tsdeltas = {}
    
        # user the dates in evaluator because we've already incurred the expense of parsing the date.
        self.tsdeltas = dict((ts.to_datetime_string(), self.evaluator.dates[i + 1] - ts) for i, ts in
                                 enumerate(self.evaluator.dates[:-1]))
        self.tsdeltas[self.evaluator.dates[-1].to_datetime_string()] = self.tsdeltas[self.evaluator.dates[-2].to_datetime_string()]  # TODO: fix this


        # prepare data - we could move some of this to elsewhere

        template_id = template.id

        # extract info about nodes & links
        self.network = network
        self.nodes = {}
        self.links = {}
        self.ttypes = {'node': {}, 'link': {}, 'network': {}}
        self.res_attrs = {}
        self.link_nodes = {}
        
        
        self.params = {} # to be defined later
        self.nparams = 0
        self.nvars = 0 
        
        ttypeattrs = {'node': {}, 'link': {}, 'network': {}}
        rtypeattrs = {'node': {}, 'link': {}, 'network': {}}

        for tt in template.types:
            ttype_name = convert_type_name(tt.name)
            resource_type = tt.resource_type.lower() # i.e., node, link, network
            self.ttypes[resource_type][ttype_name] = []
            
            ttypeattrs[resource_type][ttype_name] = [ta.attr_id for ta in tt.typeattrs] # use typeattrs to track variables/parameters

        # organize basic network information
        #features['networks'] = [network]
        for resource_type in ['node', 'link']:
            for resource in network['{}s'.format(resource_type)]:
                rtypes = list(filter(lambda x: x.template_id == template.id, resource.types))
                if not rtypes:
                    continue
                    
                if resource_type == 'node':
                    idx = resource.id
                    self.nodes[idx] = resource
                elif resource_type == 'link':
                    idx = (resource.node_1_id, resource.node_2_id)
                    self.links[idx] = resource
                    self.link_nodes[resource.id] = idx

                # a dictionary of template_type to node_id
                rtype = rtypes[-1]
                type_name = convert_type_name(rtype.name)
                if type_name not in self.ttypes[resource_type]:
                    self.ttypes[resource_type][type_name] = []
                self.ttypes[resource_type][type_name].append(idx)
                
                rtypeattrs[resource_type][resource.id] = ttypeattrs[resource_type][type_name]

                # general resource attribute information
                for ra in resource.attributes:
                    attr_id = ra.attr_id
                    #if attr_id in list(attrs.keys()) + rtypeattrs[resource_type][resource.id]:
                    if attr_id in rtypeattrs[resource_type][resource.id]:
                        self.res_attrs[ra.id] = {
                            'name': attrs[attr_id]['name'],
                            'type': resource_type,
                            'data_type': 'timeseries',
                            'is_var': ra.attr_is_var
                        }
                        
                        if ra.attr_is_var == 'N' and args.write_input:
                            self.nparams += 1
                        else:
                            self.nvars += 1

        # initialize dictionary of parameters
        self.scalars = {feature_type: {} for feature_type in ['node', 'link', 'net']}

        self.ra_node = {ra.id : node.id for node in network.nodes  for ra in node.attributes}  # res_attr to node lookup
        self.ra_link = {ra.id: link.id for link in network.links for ra in link.attributes}  # res_attr to link lookup

        #ra_net = dict() # res_attr to network lookup
        #for link in network.links:
            #for res_attr in link.attributes:
                #ra_link[res_attr.id] = link.id

        # may be useful in the future
        #self.class_lookup = {
            #'node': 'nodes',
            #'link': 'links',
            #'network': 'network'
        #}
        
        # define foresight periods & timesteps
        # NB: to be as efficient as possible within run loops, we should keep as much out of the loops as possible
        self.nruns = len(self.dates)
        foresight = args.foresight if 'foresight' in args else 'zero' # pending further development
        if foresight == 'perfect':
            self.foresight_periods = len(system.dates)
            self.save_periods = self.foresight_periods
            self.nruns = 1
        elif foresight == 'zero':
            self.foresight_periods = 1
            self.save_periods = 1
        self.ts_idx=range(self.foresight_periods)                       

    def collect_source_data(self):
        """
        This does some pre-processing to organize data for more efficient collection later.
        """
        
        tsi=0
        tsf=self.foresight_periods

        #self.scenario = scenario
        self.timeseries = {}
        self.variables = {}
        self.block_params = ['Demand', 'Priority']
        self.blocks = {'node': {}, 'link': {}}
        self.results = {}
        self.res_scens = {}
        
        self.evaluator.tsi = tsi
        self.evaluator.tsf = tsf
        
        # collect source data
        for source_id in self.scenario.source_ids:

            self.evaluator.scenario_id = source_id

            source = self.scenario.source_scenarios[source_id]

            for rs in source.resourcescenarios:
                if rs.resource_attr_id not in self.res_attrs:
                    continue # this is for a different resource type
                if self.res_attrs[rs.resource_attr_id]['is_var'] == 'Y':
                    continue # this is a dependent (output) variable

                # create a dictionary to lookup resourcescenario by resource attribute ID
                self.res_scens[rs.resource_attr_id] = rs

                # load the metadata
                metadata = json.loads(rs.value.metadata)

                # get identifiers
                if rs.resource_attr_id in self.ra_node:
                    resource_type = 'node'
                    idx = self.ra_node[rs.resource_attr_id]
                elif rs.resource_attr_id in self.ra_link:
                    resource_type = 'link'
                    idx = self.link_nodes[self.ra_link[rs.resource_attr_id]]
                else:
                    continue # network attributes don't belong in the model (at least for now)
                    #resource_type = 'net'
                    #fid = self.ra_net[rs.resource_attr_id]

                # identify as function or not
                is_function = metadata.get('use_function', 'N') == 'Y'

                # get attr name
                attr_name = self.res_attrs[rs.resource_attr_id]['name']

                # get data type
                data_type = rs.value.type

                # update data type
                self.res_attrs[rs.resource_attr_id]['data_type'] = data_type

                # default blocks
                # NB: self.block_params should be defined
                has_blocks = (attr_name in self.block_params) or metadata.get('has_blocks', 'N') == 'Y'
                blocks = [0]

                param_name = get_param_name(resource_type, attr_name)

                #value = rs.value.value
                # TODO: get fill_value from dataset/ttype (this should be user-specified)
                self.evaluator.data_type = data_type          
                try:
                    value = self.evaluator.eval_data(
                        value=rs.value,
                        do_eval=False,
                        flavor='pandas',
                        fill_value=0,
                        res_attr_id=rs.resource_attr_id
                    )
                except:
                    raise
                
                # TODO: add generic unit conversion utility here
                dimension = rs.value.dimension

                if data_type == 'scalar':
                    if param_name not in self.variables:
                        self.variables[param_name] = {}
                    
                    value = float(value) # TODO: add conversion?
                                           
                    self.variables[param_name][idx] = value

                elif data_type == 'descriptor': # this could change later
                    if param_name not in self.variables:
                        self.variables[param_name] = {}
                    self.variables[param_name][idx] = value

                elif data_type == 'timeseries':
                    values = value
                    function = None

                    if is_function:
                        function = metadata['function']
                        if len(function) == 0: # if there is no function, this will be treated as no dataset
                            continue
                        if has_blocks:
                            blocks = range(value.columns.size)

                    else:
                        values = pd.read_json(value)
                        
                        #if dimension == 'Volumetric flow rate' and not is_function:
                            #for i, datetime in enumerate(self.dates_as_string[tsi:tsf]):
                                #values.iloc[i] *= self.tsdeltas[datetime].days * self.VOLUMETRIC_FLOW_RATE_CONST

                        if has_blocks:
                            blocks = list(range(values.columns.size))

                    if param_name not in self.timeseries:
                        self.timeseries[param_name] = {}
                                            
                    self.timeseries[param_name][idx] = {
                        'data_type': data_type,
                        'values': values,
                        'function': function,
                        'has_blocks': has_blocks,
                        'dimension': dimension,
                    }

                if idx in self.blocks[resource_type]:
                    n_existing = len(self.blocks[resource_type][idx])
                    n_new = len(blocks)
                    blocks = range(max([n_existing, n_new]))
                self.blocks[resource_type][idx] = blocks

    def prepare_params(self):
        """
        Declare parameters, based on the template type.
        The result is a dictionary of all parameters for later use and extension.
        """

        for ttype in self.template.types:
            
            resource_type = ttype['resource_type']
            
            if resource_type == 'NETWORK':
                continue

            for type_attr in ttype.typeattrs:

                data_type = type_attr['data_type']

                # create a unique parameter name
                param_name = get_param_name(resource_type, type_attr['attr_name'])

                if param_name in self.params:
                    continue
          
                self.params[param_name] = {
                    'attr_name': type_attr['attr_name'],
                    'type_attr': type_attr,
                    'is_var': type_attr['is_var'],
                    'resource_type': resource_type.lower(),
                }
                            
    def setup_variations(self, variation_sets):
        """
        Add variation to all resource attributes as needed.
        There are two variations: option variations and scenario variations.
        If there is any conflict, scenario variations will replace option variations.
        """
                
        for variation_set in variation_sets:
            for (resource_type, resource_id, attr_id), variation in variation_set['variations'].items():
                attr = self.conn.attrs[resource_type][attr_id]
                param_name = get_param_name(resource_type, attr['name'])
                if resource_type == 'node':
                    idx = resource_id
                elif resource_type == 'link':
                    idx = self.link_nodes[resource_id]
                # TODO: add other resource_types
                
                # at this point, timeseries have not been assigned to variables, so these are mutually exclusive
                # the order here shouldn't matter
                variable = self.variables.get(param_name, {}).get(idx)
                timeseries = self.timeseries.get(param_name, {}).get(idx)
                if variable:
                    self.variables[param_name][idx] = perturb(self.variables[param_name][idx], variation)
                
                elif timeseries:
                    if not timeseries.get('function'): # functions will be handled by the evaluator
                        self.timeseries[param_name][idx]['values'] = perturb(self.timeseries[param_name][idx]['values'], variation)
                        
                else: # we need to add the variable to account for the variation
                    data_type = attr['dtype']
                    if data_type == 'scalar':
                        if param_name not in self.variables:
                            self.variables[param_name] = {}
                        self.variables[param_name][idx] = perturb(0, variation)
                    elif data_type == 'timeseries':

                        default_timeseries = pd.read_json(self.evaluator.default_timeseries).fillna(0)
                        # TODO: get default from attr
                        
                        if param_name not in self.variables:
                            self.variables[param_name] = {}
                        self.timeseries[param_name][idx] = {
                            'values': perturb(default_timeseries, variation),
                            'dimension': attr['dim']
                        }
                        
                    
    def init_pyomo_params(self):
        """Initialize Pyomo parameters with definitions."""
        
        for param_name, param in self.params.items():

            type_attr = param['type_attr']
            data_type = type_attr['data_type']
            resource_type = param['resource_type']
            attr_name = param['attr_name']
            
            param_definition = None
    
            initial_values = self.variables.get(param_name, None)
        
            if param['is_var'] == 'N':
                
                mutable = True  # assume all variables are mutable
                default = 0 # TODO: define in template rather than here
    
                if data_type == 'scalar':
                    param_definition = 'm.{rt}s'

                elif data_type == 'timeseries':
                    if attr_name in self.block_params:
                        param_definition = 'm.{rt}Blocks, m.TS'
                    else:
                        param_definition = 'm.{rt}s, m.TS'

                elif data_type == 'array':
                    continue  # placeholder

                else:
                    continue

                param_definition += ', default={}, mutable={}'.format(default, mutable)
                if initial_values is not None:
                    param_definition += ', initialize=initial_values'
                    # TODO: This is an opportunity for allocating memory in a Cythonized version?
    
                param_definition = param_definition.format(rt=resource_type.title())
    
            expression = 'm.{param_name} = Param({param_definition})'.format(
                param_name=param_name,
                param_definition=param_definition
            )
    
            self.params[param_name].update({
                'initial_values': initial_values,
                'expression': expression
            })
                
            
                
    def update_initial_conditions(self):
        """Update initial conditions, such as reservoir and groundwater storage."""

        # we should provide a list of pairs to map variable to initial conditions (reservoir storage, groundwater storage, etc.)
        # Storage includes both reservoirs and groundwater
        for j in self.instance.Storage:
            getattr(self.instance, 'nodeInitialStorage')[j] = getattr(self.instance, 'nodeStorage')[j,0].value

    def update_boundary_conditions(self, tsi, tsf, initialize=False):
        """Update boundary conditions. If initialize is True, this will create a variables object for use in creating the model (i.e., via init_pyomo_params). Otherwise, it will update the model instance."""
        
        dates_as_string = self.dates_as_string[tsi:tsf]
        
        for param_name, param in self.timeseries.items():
            for idx, p in param.items():
                is_function = p.get('function')
                dimension = p.get('dimension')
                if is_function:
                    self.evaluator.data_type = p['data_type']
                    self.evaluator.tsi = tsi
                    self.evaluator.tsf = tsf
                    try:
                        returncode, errormsg, df = self.evaluator.eval_function(p['function'], flavor='pandas', counter=0)
                    except:
                        raise
                                        
                else:
                    df = p['values']
                    
                    
                for j, c in enumerate(df.columns):

                    # update values variable
                    for i, datetime in enumerate(dates_as_string):

                        val = df.loc[datetime, c]
                        
                        #if is_function:
                        # TODO: use generic unit converter here (and move to evaluator)
                        if dimension == 'Volumetric flow rate':
                            val *= self.tsdeltas[datetime].days * self.VOLUMETRIC_FLOW_RATE_CONST                    
                            
                        # create key
                        key = list(idx) + [i] if type(idx) == tuple else [idx,i]
                        if p.get('has_blocks'):
                            key.insert(-1, j)
                        key = tuple(key)

                        if initialize:
                            if param_name not in self.variables:
                                self.variables[param_name] = {}
                            self.variables[param_name][key] = val

                        else:  # just update the parameter directly
                            try:
                                # TODO: replace this with explicit updates
                                getattr(self.instance, param_name)[key] = val
                            except:
                                pass # likely the variable simply doesn't exist in the model
        

    def update_internal_params(self):
        '''Update internal parameters based on calculated variables'''

        # define values based on user-defined priorities
        lowval = 100
        for idx in self.instance.nodePriority:
            getattr(self.instance, 'nodeValueDB')[idx] = lowval - (getattr(self.instance, 'nodePriority')[idx].value or lowval)
        for idx in self.instance.linkPriority:
            getattr(self.instance, 'linkValueDB')[idx] = lowval - (getattr(self.instance, 'linkPriority')[idx].value or lowval)


    def collect_results(self, timesteps, include_all=False, write_input=True):
        
        all_records = []

        # loop through all the model parameters and variables
        for p in self.instance.component_objects(Param):
            if write_input or p.name in ['nodeDemand', 'nodeObservedDelivery']:
                records = self.store(p, timesteps, is_var=False, include_all=include_all)
                all_records.extend(records)
                
        for v in self.instance.component_objects(Var):
            records = self.store(v, timesteps, is_var=True, include_all=include_all)
            all_records.extend(records)
        
        #self.recorder.record(all_records)
        # todo: fix this routine!

    def store(self, p, timesteps, is_var, include_all=None):

        if p.name not in self.results:
            self.results[p.name] = {}
            
        records = []

        # collect to results
        for v in p.values():  # loop through parameter values
            try:
                idx = v.index()
            except:
                continue
            
            rt = p.name[:4]
            if is_var:

                # this assumes that all decision variables are time series
                # TODO: Verify this assumption
                res_idx = len(idx)==2 and idx[0] or idx[:-1]
                time_idx = idx[-1]
            else:
                res_idx = type(idx)==int and idx or idx[:-1]
                time_idx = type(idx) != int and idx[-1]
        
            if time_idx is not False and time_idx==0 or include_all:  # index[-1] is time
                
                if rt=='node':
                    resource = self.nodes.get(res_idx if type(res_idx)==int else res_idx[0])
                elif rt=='link':
                    resource = self.links.get(res_idx[:2])
                else:
                    rt = 'network'
                    resource = self.network
                
                if res_idx not in self.results[p.name]:  # idx[:-1] is node/link + block, if any
                    self.results[p.name][res_idx] = OrderedDict()
                    
                timestamp = timesteps[time_idx]
                val = v.value if v.value is None else float(v.value)
                    
                # TODO: We should do one or the other (or both?)
                # 1) save results for writing later back to Hydra Platform
                self.results[p.name][res_idx][timestamp] = val
        
                # or 2) save single time step results to write back via real-time recorder
                # NB: we can make this more efficient through better DB design (follow Hyrda Platform schema?)
                #records.append({
                    #'network_id': self.network.id,
                    #'option': self.option.name,
                    #'scenario': self.scenario.name,
                    #'resource_type': rt,
                    #'resource': resource.name,
                    #'attribute': p.name,
                    #'timestamp': timestamp,
                    #'value': val
                #});
                
        return records

    def save_results(self):
        
        if self.scenario.reporter:
            self.scenario.reporter.report(action='save', saved=0)
        
        if self.args.destination == 'source':
            self.save_results_to_source()
        elif self.args.destination == 'aws_s3':
            self.save_results_to_s3()
        return
        
    def save_results_to_source(self):

        result_scenario = self.scenarios.get(self.scenario.name)
        if result_scenario and result_scenario.id not in self.scenario.source_ids:
            self.conn.call('purge_scenario', {'scenario_id': result_scenario.id})
        result_scenario = self.conn.call('add_scenario',
                       {'network_id': self.network.id, 'scen': {
                           'id': None,
                           'name': self.scenario.name,
                           'description': '',
                           'network_id': self.network.id,
                           'layout': {'class': 'results', 'sources': self.scenario.base_ids, 'tags': self.scenario.tags}
                       }})


        # save variable data to database
        res_scens = []
        mb = 0
        res_names = {}
        
        try:
            count = 1
            pcount = 1
            nparams = len(self.results)
            for pname, param_values in self.results.items():
                #if self.args.debug and pcount == 5:
                    #break
                pcount += 1
                if pname not in self.params:
                    continue  # it's probably an internal variable/parameter
                rt = self.params[pname]['resource_type']
                ta = self.params[pname]['type_attr']
                attr_id = ta['attr_id']
                attr = self.conn.attrs[rt][attr_id]
    
                # reorganize values as stored by Pyomo to resource attributes
                # pid = Pyomo resource attribute id
                dataset_values = {}
                for idx, values in param_values.items():
                    idx = type(idx)==tuple and list(idx) or [idx] # needed to concatenate with the attribute name
                    if rt=='node':
                        n = 1
                        pid = (idx[0], ta['attr_name'])
                        res_name = self.nodes[pid[0]]['name']
                    elif rt == 'link':
                        n = 2
                        pid = (idx[0], idx[1], ta['attr_name'])
                        res_name = self.links[pid[:n]]['name']
                    else:
                        # TODO: Include Network resource data here
                        continue
    
                    if pid not in self.conn.res_attr_lookup[rt]:
                        continue
    
                    block = 0 if len(idx)==n else idx[n]
                    if pid not in dataset_values:
                        dataset_values[pid] = {}
                    dataset_values[pid][str(block)] = values
                    if pid not in res_names: res_names[pid] = res_name
    
                # create datasets from values
                for pid, dataset_value in dataset_values.items():
    
                    # define the dataset value
                    value = json.dumps(OrderedDict(sorted(dataset_value.items())))
    
                    # create the resource scenario (dataset attached to a specific resource attribute)
                    rs = {
                        'resource_attr_id': self.conn.res_attr_lookup[rt][pid],
                        'attr_id': attr_id,
                        'dataset_id': None,
                        'value': {
                            'type': attr['dtype'],
                            'name': '{} - {} - {} [{}]'.format(self.network.name, res_names[pid], attr['name'], self.scenario.name),
                            'unit': attr['unit'],
                            'dimension': attr['dim'],
                            'value': value
                        }
                    }
                    res_scens.append(rs)
                    mb += len(value.encode())*1.1/1e6 # large factor of safety
            
                    if mb > 10 or count==nparams:
                        self.conn.call('update_resourcedata', {
                            'scenario_id': result_scenario['id'],
                            'resource_scenarios': res_scens[:-1]
                        })
                        if count % 10 == 0 or pcount==nparams:
                            if self.scenario.reporter:
                                self.scenario.reporter.report(action='save', saved=round(count/(self.nparams+self.nvars)*100) )
                        count += len(res_scens)
                        
                        # purge just-uploaded scenarios
                        res_scens = res_scens[-1:]
                        mb = 0
                        
            # upload the last remaining resource scenarios
            self.conn.call('update_resourcedata', {
                'scenario_id': result_scenario['id'],
                'resource_scenarios': res_scens
            })
            if self.scenario.reporter:
                self.scenario.reporter.report(action='save', saved=round(count/(self.nparams+self.nvars)*100) )                   
                                                                
        except:
            msg = 'ERROR: Results could not be saved.'
            #self.logd.info(msg)
            if self.scenario.reporter:
                self.scenario.reporter.report(action='error', message=msg)
            if self.session:
                self.session.leave()
            sys.exit(0)

    
    def save_results_to_s3(self):
        
        import boto3
        #s3 = boto3.client('s3', aws_access_key_id=args.AWS_ACCESS_KEY_ID, aws_secret_access_key=args.AWS_SECRET_ACCESS_KEY)
        s3 = boto3.client('s3')

        result_scenario = self.scenarios.get(self.scenario.name)

        # save variable data to database
        res_scens = []
        res_names = {}
        try:
            count = 1
            pcount = 1
            nparams = len(self.results)
            for pname, param_values in self.results.items():
                pcount += 1
                if pname not in self.params:
                    continue  # it's probably an internal variable/parameter
                rt = self.params[pname]['resource_type']
                ta = self.params[pname]['type_attr']
                attr_id = ta['attr_id']
                attr = self.conn.attrs[rt][attr_id]
    
                # reorganize values as stored by Pyomo to resource attributes
                # pid = Pyomo resource attribute id                             
                # create datasets from values
                df_all = pd.DataFrame()
                
                #dataset_values = {}
                for idx, values in param_values.items():
                    idx = type(idx)==tuple and list(idx) or [idx] # needed to concatenate with the attribute name
                    if rt=='node':
                        n = 1
                        pid = (idx[0], ta['attr_name'])
                        res_name = self.nodes[pid[0]]['name']
                    elif rt == 'link':
                        n = 2
                        pid = (idx[0], idx[1], ta['attr_name'])
                        res_name = self.links[pid[:n]]['name']
                    else:
                        # TODO: Include Network resource data here
                        continue
    
                    if pid not in self.conn.res_attr_lookup[rt]:
                        continue
                    
                    has_blocks = ta.properties.get('has_blocks') \
                        or rt == 'node' and len(idx) == 2 \
                        or rt=='link' and len(idx) == 3
                    
                    if has_blocks:
                        block = 0 if len(idx)==n else idx[n]
                        df = pd.DataFrame.from_dict({(res_name, block): values})
                    else:
                        df = pd.DataFrame.from_dict({res_name: values})
                    df_all = pd.concat([df_all, df], axis=1)
               
                content = df_all.to_csv().encode()
                path = 'results/P{}/N{}/{}/{}/{}.csv'.format(self.network.project_id, self.network.id, self.args.start_time, self.scenario.name, pname)
                s3.put_object(Body=content, Bucket='openagua.org', Key=path)  
                
                if count % 10 == 0 or pcount==nparams:
                    if self.scenario.reporter:
                        self.scenario.reporter.report(action='save', progress=100, saved=round(count/(self.nparams+self.nvars)*100) )
                count += 1
                
        except:
            msg = 'ERROR: Results could not be saved.'
            #self.logd.info(msg)
            if self.scenario.reporter:
                self.scenario.reporter.report(action='error', message=msg)
            if self.session:
                self.session.leave()
            sys.exit(0)

