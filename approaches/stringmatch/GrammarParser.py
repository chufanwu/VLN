import json
import nltk
import random

from code_v6.utils import buildCategoryVocabulary


class GrammarParser():
    def __init__(self, configPath):

        self.categoryVocPath = '/Users/dongqiyuan/Desktop/GoalMap/data/category_vocab.txt'
        self.categoryVoc = buildCategoryVocabulary(self.categoryVocPath)
        self.grammarDict = {}

        self.objectDetectMark = ['until', 'to', 'that', 'when', 'if', 'before', 'but', 'past', 'passed','by', 'with',
                                 'through',
                                 'so', 'as', 'between', 'at', 'into', 'where', 'near', 'towards', 'toward', 'away','using']

        with open(configPath, 'r') as fp:
            config = json.load(fp)
        config = config[0]

        for (verb, prep) in config.items():
            # dict for preposition
            d_prep = {}
            for p in prep:
                p_sep = p.split('/')
                for i_p_sep in p_sep:
                    d_prep[i_p_sep] = p

            # add each verb with such dict
            verb_sep = verb.split('/')
            for i_verb_sep in verb_sep:
                self.grammarDict[i_verb_sep] = d_prep

        self.verbs = [v for (v, p) in self.grammarDict.items()]

    def parse(self, instruction):
        instruction = self.instruction_prepocess(instruction)
        verb = self.verb_exist(instruction)
        if not verb:
            print 'Parse End.'
            return False

        addition1, verbInst = instruction.split(str(verb) + ' ')
        addition1, verbInst = addition1.strip(), verbInst.strip()

        prep, addition2, afterPrepInst = self.prep_exist(verbInst, verb)
        if prep is False:
            print 'Parse End.'
            return False

        objects, addition3 = self.object_detect(afterPrepInst)

        if addition1 != '':
            addition1 += ' '
        verb += ' '
        if addition2 != '':
            addition2 += ' '
        prep += ' '
        if objects != '':
            objects += ' '
        if addition3 != '':
            addition3 += ' '

        pattern_dis = ' ' * len(addition1) + str(verb) + ' ' * len(addition2) + prep + ' ' * len(objects) + ' ' * len(
            addition3)
        objects_dis = ' ' * len(addition1) + ' ' * len(verb) + ' ' * len(addition2) + ' ' * len(
            prep) + objects + ' ' * len(addition3)
        addition1_dis = addition1 + ' ' * len(verb) + ' ' * len(addition2) + ' ' * len(prep) + ' ' * len(
            objects) + ' ' * len(addition3)
        addition2_dis = ' ' * len(addition1) + ' ' * len(verb) + addition2 + ' ' * len(prep) + ' ' * len(
            objects) + ' ' * len(addition3)
        addition3_dis = ' ' * len(addition1) + ' ' * len(verb) + ' ' * len(addition2) + ' ' * len(prep) + ' ' * len(
            objects) + addition3

        s = '#' * 100 + '\n'
        s += 'Instruction:        %s\n\n' % instruction
        s += '    Pattern:        %s\n' % pattern_dis
        s += '    Objects:        %s\n' % objects_dis
        s += ' Addition-1:        %s\n' % addition1_dis
        s += ' Addition-2:        %s\n' % addition2_dis
        s += ' Addition-3:        %s\n' % addition3_dis
        s += '#' * 100

        print s

        return True

    def verb_exist(self, instruction):
        exist = [v for v in self.verbs if v in instruction.split(' ')]

        if not exist:
            print 'Error. Instruction        ### %s ###      has no verb.' % instruction
            return False

        elif len(exist) > 1:
            print 'Error. Instruction       ### %s ###      has more than one verb.' % instruction
            return False

        else:
            return exist[0]

    def prep_exist(self, verbInst, verb):

        verbInst_padding = ' ' + verbInst + ' '

        prep_candidates = self.grammarDict[verb].keys()
        exist = [(p,verbInst_padding.index(p)) for p in prep_candidates if ' ' + p + ' ' in verbInst_padding]
        # put the longer prep first
        exist.sort(key=lambda x:x[0].count(' '))
        exist.reverse()

        if exist:
            if exist[0][0].count(' ') == 0:
                exist.sort(key=lambda x: x[1])

        # if verb may has no preposition
        if self.grammarDict[verb].has_key('no') and not exist:
            if self.no_prep_test(verbInst):
                addition2 = ''
                afterPrepInst = verbInst
                return '', addition2, afterPrepInst.strip()
            else:
                print 'Error. Instruction       ### %s ###      has no matched preposition with verb    ### %s ###.     ' % (
                    verbInst, verb)
                return False, False, False


        elif not exist:
            print 'Error. Instruction       ### %s ###      has no matched preposition with verb    ### %s ###.     ' % (
            verbInst, verb)
            return False, False, False

        else:
            # one word prep
            if len(exist[0][0].split(' ')) == 1:
                prep = exist[0][0]
                verbInst_sep = verbInst.split(' ')

                # the first position where prep appears
                idx = verbInst_sep.index(prep)

                addition2 = ' '.join(verbInst_sep[:idx])
                afterPrepInst = ' '.join(verbInst_sep[idx + 1:])

            # mutiple word prep
            else:
                prep = exist[0][0]
                prep_sep = exist[0][0].split(' ')
                verbInst_sep = verbInst.split(' ')

                # the first position where prep appears
                idx1 = verbInst_sep.index(prep_sep[0])
                idx2 = verbInst_sep.index(prep_sep[-1])

                addition2 = ' '.join(verbInst_sep[:idx1])
                afterPrepInst = ' '.join(verbInst_sep[idx2 + 1:])

            return prep, addition2.strip(), afterPrepInst.strip()

    def object_detect(self, afterPrepInst):
        afterPrepInst_sep = afterPrepInst.split(' ')

        idx = None
        for marker in self.objectDetectMark:
            if marker in afterPrepInst_sep:
                idx = afterPrepInst_sep.index(marker)
                break

        if idx is not None:
            objects = ' '.join(afterPrepInst_sep[:idx])
            addition3 = ' '.join(afterPrepInst_sep[idx:])
        else:
            objects = ' '.join(afterPrepInst_sep)
            addition3 = ''

        return objects.strip(), addition3.strip()

    def no_prep_test(self,verbInst):
        next_word = verbInst.split(' ')[0]
        n, v = nltk.pos_tag([next_word])[0]

        if next_word in self.categoryVoc or ('NN' in v):
            return True

        if len(verbInst.split(' ')) > 1:
            next_word = verbInst.split(' ')[1]
            n, v = nltk.pos_tag([next_word])[0]
            if next_word in self.categoryVoc or ('NN' in v):
                return True

        return False


    def instruction_prepocess(self, instruction):
        inst_sep = instruction.split(' ')
        # convert all letter to lowercase
        inst_sep = [word.lower() for word in inst_sep]

        # remove 'the'
        while 'the' in inst_sep:
            inst_sep.remove('the')

        # remove ' '
        while ' ' in inst_sep:
            inst_sep.remove(' ')

        # remove ''
        while '' in inst_sep:
            inst_sep.remove('')

        inst = ' '.join(inst_sep)

        return inst


configPath = '/Users/dongqiyuan/Desktop/GoalMap/config_all.json'
parser = GrammarParser(configPath)

inst = 'make a sharp turn right'
parser.parse(inst)
# test = True
test = False

if test:
    instPath = '/Users/dongqiyuan/Desktop/GoalMap/data/only_short_instructions.txt'
    instructions = []
    with open(instPath, 'r') as fp:
        for line in fp:
            if '\n' in line:
                line = line[:-1]
            instructions.append(line)

    failRecordPath = '/Users/dongqiyuan/Desktop/GoalMap/data/failed_instruction.txt'
    failRecord = []
    success_cnt = 0
    for i in instructions:
        result = False
        try:
            result = parser.parse(i)
        except Exception, e:
            pass

        if result:
            success_cnt += 1
        else:
            # flag = False
            # for v in parser.verbs:
            #     if ' ' + v + ' ' in i:
            #         flag = True
            # if flag:
            #     failRecord.append(i)
            failRecord.append(i)

    print 'Success: %d / %d' % (success_cnt, len(instructions))

    with open(failRecordPath, 'w') as fp:
        for i in failRecord:
            fp.write(i + '\n')
