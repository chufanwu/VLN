import json
import os
import nltk
from nltk.stem import WordNetLemmatizer


from relative_location_match import RelativeLocationMatch
from action_match import ActionMatch
from relation_match import RelationMatch

def clean_sentence(s):
    def remove_char(s, c):
        while c in s:
            s = s.replace(c, '')
        return s
    s = remove_char(s, '(')
    s = remove_char(s, ')')

    FLAG = ['OBJ','REGION','LOCATION']
    for f in FLAG:
        s = s.replace(f,f.lower())

    # remove 'A' to 'Z'
    for i in range(65, 91):
        s = remove_char(s, chr(i))

    s = s.split(' ')
    while '' in s:
        s.remove('')
    while ' ' in s:
        s.remove(' ')

    for idx in range(len(s)):
        for f in FLAG:
            if f.lower() in s[idx]:
                s[idx] = s[idx].replace(f.lower(),f)
                break

    s = ' '.join(s)



    return s


def find_type(s, target):
    result = []

    def find_type_once(s, target):
        oriStr = s
        idx = s.index(target)
        s = s[idx + len(target):]

        bracketCnt = 1
        resultStr = ''
        while bracketCnt != 0:
            c = s[0]
            resultStr += c
            s = s[1:]

            if c == '(':
                bracketCnt += 1
            elif c == ')':
                bracketCnt -= 1

        resultStr = resultStr[:-1]
        nextStr = oriStr[:idx] + target.lower() + oriStr[idx+len(target):]

        return resultStr.strip(), nextStr

    while target in s:
        resultStr, s = find_type_once(s,target)
        result.append(resultStr)

    return result

def load_parser_result(path):
    d = {}
    with open(path,'r') as fp:
        d = json.load(fp)
    return d

def split_instruction(s):
    verb_list = ['walk', 'stop', 'enter', 'then', 'start', 'turn', 'exit', 'move', 'continue', 'wait', 'head', 'stand',
                 'exit', 'when', 'step', 'climb', 'make', 'follow']
    verb_list_exempt = ['painting', 'left', 'right', 'is', 'are', 'railing', 'the', 'living']
    prep_list = ['into', 'towards', 'up', 'out', 'down', 'across', 'pass', 'to', 'straight', 'through', 'past', 'right',
                 'toward']
    comma_sep_verb_list = ['walk', 'stop', 'enter', 'turn', 'exit', 'move', 'continue', 'wait', 'head', 'stand', 'exit',
                           'step', 'climb', 'make', 'follow', 'go']

    record = {}
    record['original instruction'] = s
    record['split instructions'] = []
    for inst_end in s.split('.'):

        # do not split at comma if there is no verb
        inst_commasep = inst_end.split(',')
        if len(inst_commasep) >= 2:
            first = inst_commasep[0].split(' ')
            while '' in first:
                first.remove('')
            while ' ' in first:
                first.remove(' ')

            no_verb = True
            for (n, v) in nltk.pos_tag(first):
                if 'VB' in v or n.lower() in comma_sep_verb_list:
                    no_verb = False
            if no_verb:
                inst_commasep[1] = inst_commasep[0] + ' ' + inst_commasep[1]
                inst_commasep.pop(0)

        for inst_comma in inst_commasep:
            flag_and = []

            inst_sep = inst_comma.split(' ')
            inst_sep = [i.lower() for i in inst_sep]
            while '' in inst_sep:
                inst_sep.remove('')
            while ' ' in inst_sep:
                inst_sep.remove(' ')

            if inst_sep:
                if inst_sep[0] in ['and', 'then']:
                    inst_sep.pop(0)
                inst_comma = ' '.join(inst_sep)

            for idx in range(len(inst_sep)):
                if inst_sep[idx] in ['and', 'then'] and idx != 0 and idx != (len(inst_sep) - 1):
                    prev_word = inst_sep[idx - 1]
                    next_word = inst_sep[idx + 1]
                    words = [prev_word, next_word]

                    for (n, v) in nltk.pos_tag(words):
                        if (('VB' in v or n in verb_list) and (n not in verb_list_exempt) and ('ing' not in n)) or (
                            n in prep_list):
                            if next_word in prep_list:
                                inst_sep.insert(idx + 1, 'walk')
                            flag_and = True
                            inst_sep[idx] = 'SPLIT'
                            break

            if flag_and:
                inst_new = ' '.join(inst_sep)
                inst_sep_by_and = inst_new.split('SPLIT')
                for inst_ in inst_sep_by_and:
                    if inst_ != u' ':
                        inst_save = inst_.strip()
                        record['split instructions'].append(inst_save)

            else:
                if inst_comma != u' ':
                    inst_comma = inst_comma.strip()
                    record['split instructions'].append(inst_comma)

    # deal with 'to'
    to_verbs = ['go', 'walk', 'turn', 'exit', 'enter', 'make', 'take', 'move', 'head', 'stop', 'wait', 'continue']
    to_verbs_exempt = ['continue', 'need']

    new_split = []
    for i in record['split instructions']:
        i_padding = ' ' + i + ' '
        sep_flag = False
        for v in to_verbs:
            if 'to ' + v in i_padding:
                exempt_flag = False
                # test verbs exempt
                for v_e in to_verbs_exempt:
                    if v_e + ' to' in i_padding:
                        exempt_flag = True
                if not exempt_flag:
                    inst_1, inst_2 = i_padding.split('to ' + v)
                    inst_2 = v + inst_2
                    new_split.append(inst_1.strip())
                    new_split.append(inst_2.strip())
                    sep_flag = True
                    break
        if not sep_flag:
            new_split.append(i)

    return new_split

s = 'turn around and exit the room through the wood doors. turn left and enter hallway. turn into the first door on your left. stop once you enter the room facing the sink on the back wall.'
r = split_instruction(s)
print r


