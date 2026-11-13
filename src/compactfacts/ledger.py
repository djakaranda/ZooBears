"""Optional cross-call state with provenance. Separate from per-call predictions."""
import copy

SINGLE={'service_state','tariff','ticket_status'}


class TimelineLedger:
    def __init__(self):self.accounts={}

    def ingest(self,call,prediction):
        account=(call.get('series') or {}).get('account_id') or call['call_id'].split(':')[0]
        objects=self.accounts.setdefault(account,{})
        for raw in prediction['facts']:
            fact=copy.deepcopy(raw);obj=fact['object'];oid=obj['id']
            state=objects.setdefault(oid,{'object':obj.copy(),'history':[],'current':{}})
            fact['source']['call_id']=call['call_id']
            attr=fact['attribute'];old=state['current'].get(attr)
            if attr in SINGLE and fact.get('polarity',1)>0:
                if fact['valid_to'] is not None:
                    if old and old['value']==fact['value'] and fact['valid_to']>=old['valid_from']:
                        old['valid_to']=fact['valid_to'];state['current'].pop(attr,None)
                elif old is None or fact['valid_from']>=old['valid_from']:
                    if old and old['value']!=fact['value'] and old['valid_to'] is None:
                        old['valid_to']=fact['valid_from']
                    state['current'][attr]=fact
            state['history'].append(fact)

    def snapshot(self):return copy.deepcopy({'accounts':self.accounts})
