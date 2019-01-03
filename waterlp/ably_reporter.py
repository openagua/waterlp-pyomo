from os import environ
from ably import AblyRest

# def get_ably_token(token_request=None):
#
#     token = Auth.request_token(token_request)
#     return None


class AblyReporter(object):
    
    def __init__(self, args, post_reporter, token=None):
        self.args = args
        self.post_reporter = post_reporter
        channel_name = u'com.openagua.update_s{}n{}'.format(args.source_id, args.network_id)
        if token:
            rest = AblyRest(token=token)
        else:
            rest = AblyRest(environ.get('ABLY_API_KEY'))
        self.channel = rest.channels.get(channel_name)
        self.updater = None
        
    # subscribe to actions
    def on_action(self, msg):
        action = msg['action']
        if action == 'stopall' or msg['action'] == 'stop' and msg['sid'] == args.unique_id:
            self.report(action='stop', progress=current_step / total_steps * 100)
        elif action == 'pause':
            paused = True
            self.report(action='pause', progress=current_step / total_steps * 100)
        elif action == 'resume':
            paused = False
            self.report(action='resume')
        
    # publish updates
    def report(self, action, **payload):
        if self.updater:
            payload = self.updater(action=action, **payload)
        if action in ['step', 'save']:
            self.channel.publish(action, payload)
        else:
            if self.post_reporter:
                self.post_reporter.report(**payload)
            return
            
        if action in ['done', 'error']:
            return