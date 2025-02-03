import TransformerMTGP.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import job_creation
import agent_machine
import agent_workcenter
import sequencing
import routing
import TransformerMTGP.niching.RoutingPhenoCharacterisation as RoutingPhenoCharacterisation
import TransformerMTGP.niching.SequencingPhenoCharacterisation as SequencingPhenoCharacterisation
import simpy
# import random

class shopfloor_niching:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, routing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] # added by mengxu

        # for niching by mengxu 2023.10.18
        # todo: need to change to GP evolved rule later 2023.10.18
        self.sequencingDecisionSituations = []
        self.routingDecisionSituations = []
        # phenoCharacterisation = []
        # referenceSequencingRule = "TIS"
        # sequencingPhenoCharacterisation = SequencingPhenoCharacterisation.SequencingPhenoCharacterisation(referenceSequencingRule)
        # referenceRoutingRule = "WIQ"
        # routingPhenoCharacterisation = RoutingPhenoCharacterisation.RoutingPhenoCharacterisation(referenceRoutingRule)


        # self.sequencingTree = sequencing_tree
        # self.routingTree = routing_tree
        # self.routingRule = kwargs['tree_routing']
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        #print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        if 'seed' in kwargs:
            # self.job_creator = job_creation.creation \
            #     (self.env, self.span, self.m_list, self.wc_list, [5, 25], 2, 0.9, random_seed=True, ifPrint = self.ifPrint)
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) #ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5, 25], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            wc.setJobRoutingTree(routing_tree)
            wc.setGetDecisionSituation(self.routingDecisionSituations)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)
            m.setGetDecisionSituation(self.sequencingDecisionSituations)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            # if 'tree_sequencing' in kwargs:
            #     print(str(kwargs['tree_sequencing'])) #add by mengxu to check if this is right! 2022.10.15
            #     order = "m.tree_sequencing = " + str(kwargs['tree_sequencing'])
            #     try:
            #         exec(order)
            #     except:
            #         if self.ifPrint:
            #             print("Rule assigned to machine {} is invalid !".format(m.m_idx))
            #         raise Exception
            if self.ifPrint:
                print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                # order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception

        # check if need to reset routing rule
        if 'routing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception

        # specify the architecture of DRL
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()

    def getDecisionSituations(self):
        decisionSituations = []
        if len(self.sequencingDecisionSituations) < 20:
            print("Error in get enough number of self.sequencingDecisionSituations)")

        if len(self.routingDecisionSituations) < 20:
            print("Error in get enough number of self.routingDecisionSituations)")

        np.random.shuffle(self.sequencingDecisionSituations)
        subset_sequencingDecisionSituations = []
        tryTimes = 0
        while len(subset_sequencingDecisionSituations) < 20 and tryTimes < len(self.sequencingDecisionSituations):
            if len(self.sequencingDecisionSituations[tryTimes].getData()[3]) == 3: #todo: need to decide this number always can not get 7 by mengxu 2023.10.18
                subset_sequencingDecisionSituations.append(self.sequencingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1

        print("tryTimes for sequencing = " + str(tryTimes))
        print("size for sequencing = " + str(len(subset_sequencingDecisionSituations)))

        np.random.shuffle(self.routingDecisionSituations)
        subset_routingDecisionSituations = []
        tryTimes = 0
        while len(subset_routingDecisionSituations) < 20 and tryTimes < len(self.routingDecisionSituations):
            if len(self.routingDecisionSituations[tryTimes].getData()[1]) == 3:
                subset_routingDecisionSituations.append(self.routingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1
        print("tryTimes for routing = " + str(tryTimes))
        print("size for routing = " + str(len(subset_routingDecisionSituations)))

        decisionSituations.append(subset_sequencingDecisionSituations)
        decisionSituations.append(subset_routingDecisionSituations)
        return decisionSituations

class shopfloor_niching_seq:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] # added by mengxu

        # todo: need to change to GP evolved rule later 2023.10.18
        self.sequencingDecisionSituations = []
        self.routingDecisionSituations = []

        # self.sequencingTree = sequencing_tree
        # self.routingTree = routing_tree
        # self.routingRule = kwargs['tree_routing']
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        #print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        if 'seed' in kwargs:
            # self.job_creator = job_creation.creation \
            #     (self.env, self.span, self.m_list, self.wc_list, [5, 25], 2, 0.9, random_seed=True, ifPrint = self.ifPrint)
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) #ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)
            m.setGetDecisionSituation(self.sequencingDecisionSituations)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            # if 'tree_sequencing' in kwargs:
            #     print(str(kwargs['tree_sequencing'])) #add by mengxu to check if this is right! 2022.10.15
            #     order = "m.tree_sequencing = " + str(kwargs['tree_sequencing'])
            #     try:
            #         exec(order)
            #     except:
            #         if self.ifPrint:
            #             print("Rule assigned to machine {} is invalid !".format(m.m_idx))
            #         raise Exception
            if self.ifPrint:
                print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                # order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception

        # check if need to reset routing rule
        if 'routing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception

        # specify the architecture of DRL
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()

    def getDecisionSituations(self):
        decisionSituations = []
        if len(self.sequencingDecisionSituations) < 20:
            print("Error in get enough number of self.sequencingDecisionSituations)")

        np.random.shuffle(self.sequencingDecisionSituations)
        subset_sequencingDecisionSituations = []
        tryTimes = 0
        while len(subset_sequencingDecisionSituations) < 20 and tryTimes < len(self.sequencingDecisionSituations):
            if len(self.sequencingDecisionSituations[tryTimes].getData()[3]) == 3: #todo: need to decide this number always can not get 7 by mengxu 2023.10.18
                subset_sequencingDecisionSituations.append(self.sequencingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1

        print("tryTimes for sequencing = " + str(tryTimes))
        print("size for sequencing = " + str(len(subset_sequencingDecisionSituations)))

        decisionSituations.append(subset_sequencingDecisionSituations)
        return decisionSituations

class niching_clear:
    def __init__(self, radius, capacity, **kwargs):
        self.radius = radius
        self.capacity = capacity
        self.decisionSituations = []
        self.phenotypic_characristics = []


    def initial_phenoCharacterisation(self, individual):
        # create the environment instance for simulation
        self.phenotypic_characristics = []
        env = simpy.Environment()
        dataset_name = 'HH'
        # create the shop floor instance
        rule_R = 'GP_evolve_R'
        rule_S = 'GP_evolve_S'
        # np.random.seed(seed)
        seed = 777777
        span = 100000
        m_no = 9
        wc_no = 3
        if len(individual) == 2:
            spf = shopfloor_niching(env, span, m_no, wc_no, individual[0], individual[1], routing_rule=rule_R,
                            sequencing_rule=rule_S,seed=seed, ifPrint=False, dataset_name=dataset_name)
            spf.simulation()
            self.decisionSituations = spf.getDecisionSituations()
            sequencingPhenoCharacterisation = SequencingPhenoCharacterisation.SequencingPhenoCharacterisation(individual[0], self.decisionSituations[0])
            routingPhenoCharacterisation = RoutingPhenoCharacterisation.RoutingPhenoCharacterisation(individual[1], self.decisionSituations[1])
            self.phenotypic_characristics.append(sequencingPhenoCharacterisation)
            self.phenotypic_characristics.append(routingPhenoCharacterisation)
        else:
            # this is for only evolve sequencing rule
            rule_R = 'EA'
            spf = shopfloor_niching_seq(env, span, m_no, wc_no, individual[0], routing_rule=rule_R,
                                    sequencing_rule=rule_S, seed=seed, ifPrint=False, dataset_name=dataset_name)
            spf.simulation()
            self.decisionSituations = spf.getDecisionSituations()
            sequencingPhenoCharacterisation = SequencingPhenoCharacterisation.SequencingPhenoCharacterisation(
                individual[0], self.decisionSituations[0])
            self.phenotypic_characristics.append(sequencingPhenoCharacterisation)


    def calculate_phenoCharacterisation(self, individual):
        # create the environment instance for simulation
        if len(individual) == 2:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])
            self.phenotypic_characristics[1].setReferenceRule(individual[1])
        else:
            self.phenotypic_characristics[0].setReferenceRule(individual[0])


    # def clearPopulation(self,toolbox,population):
    #     if len(population[0]) == 2:
    #         clearedInds = 0
    #         phenotypic_characristics_pop = []
    #         sorted_pop = self.sortPopulation(toolbox, population)
    #         isCleared_pop = []
    #         # calculate the PC of all individuals in population
    #         for idx in range(len(sorted_pop)):
    #             ind = sorted_pop[idx]
    #             sequencing_charList = self.phenotypic_characristics[0].characterise(ind[0])
    #             routing_charList = self.phenotypic_characristics[1].characterise(ind[1])
    #             all_charList = []
    #             for char in sequencing_charList:
    #                 all_charList.append(char)
    #             for char in routing_charList:
    #                 all_charList.append(char)
    #             phenotypic_characristics_pop.append(all_charList)
    #             isCleared_pop.append(False)
    #
    #         #clear this population
    #         for idx in range(len(sorted_pop)):
    #             if isCleared_pop[idx]:
    #                 continue
    #
    #             numWinners = 1
    #             for idy in range(idx+1, len(sorted_pop)):
    #                 if isCleared_pop[idy]:
    #                     continue
    #
    #                 distance = self.phenotypic_characristics[0].distance(phenotypic_characristics_pop[idx], phenotypic_characristics_pop[idy])
    #                 if distance > self.radius:
    #                     continue
    #
    #                 if numWinners < self.capacity:
    #                     numWinners = numWinners + 1
    #                 else:
    #                     isCleared_pop[idy] = True
    #                     len_fitness_values = len(sorted_pop[idy].fitness.values)
    #                     bad_fitness = [np.Infinity for i in range(len_fitness_values)]
    #                     sorted_pop[idy].fitness.values = bad_fitness
    #                     clearedInds = clearedInds + 1
    #
    #         print("Cleared number by niching: " + str(clearedInds))
    #     else:
    #         clearedInds = 0
    #         phenotypic_characristics_pop = []
    #         sorted_pop = self.sortPopulation(toolbox, population)
    #         isCleared_pop = []
    #         # calculate the PC of all individuals in population
    #         for idx in range(len(sorted_pop)):
    #             ind = sorted_pop[idx]
    #             sequencing_charList = self.phenotypic_characristics[0].characterise(ind[0])
    #             all_charList = []
    #             for char in sequencing_charList:
    #                 all_charList.append(char)
    #             phenotypic_characristics_pop.append(all_charList)
    #             isCleared_pop.append(False)
    #
    #         # clear this population
    #         for idx in range(len(sorted_pop)):
    #             if isCleared_pop[idx]:
    #                 continue
    #
    #             numWinners = 1
    #             for idy in range(idx + 1, len(sorted_pop)):
    #                 if isCleared_pop[idy]:
    #                     continue
    #
    #                 distance = self.phenotypic_characristics[0].distance(phenotypic_characristics_pop[idx],
    #                                                                      phenotypic_characristics_pop[idy])
    #                 if distance > self.radius:
    #                     continue
    #
    #                 if numWinners < self.capacity:
    #                     numWinners = numWinners + 1
    #                 else:
    #                     isCleared_pop[idy] = True
    #                     len_fitness_values = len(sorted_pop[idy].fitness.values)
    #                     bad_fitness = [np.Infinity for i in range(len_fitness_values)]
    #                     sorted_pop[idy].fitness.values = bad_fitness
    #                     clearedInds = clearedInds + 1
    #
    #         print("Cleared number by niching: " + str(clearedInds))
    #     return sorted_pop

    def clearPopulation(self,toolbox,population):
        if len(population[0]) == 2:
            clearedInds = 0
            phenotypic_characristics_pop = []
            sorted_pop = self.sortPopulation(toolbox, population)
            isCleared_pop = []
            # calculate the PC of all individuals in population
            for idx in range(len(sorted_pop)):
                ind = sorted_pop[idx]
                sequencing_charList = self.phenotypic_characristics[0].characterise(ind[0])
                routing_charList = self.phenotypic_characristics[1].characterise(ind[1])
                all_charList = []
                # all_charList.append(sequencing_charList)
                # all_charList.append(routing_charList)
                for ref in sequencing_charList:
                    all_charList.append(ref)
                for ref in routing_charList:
                    all_charList.append(ref)
                phenotypic_characristics_pop.append(all_charList)
                isCleared_pop.append(False)

            #clear this population
            for idx in range(len(sorted_pop)):
                if isCleared_pop[idx]:
                    continue

                numWinners = 1
                for idy in range(idx+1, len(sorted_pop)):
                    if isCleared_pop[idy]:
                        continue

                    distance = self.phenotypic_characristics[0].distance(
                        phenotypic_characristics_pop[idx], phenotypic_characristics_pop[idy])
                    if distance > self.radius:
                        continue

                    # distance_sequencing = self.phenotypic_characristics[0].distance(
                    #     phenotypic_characristics_pop[idx][0], phenotypic_characristics_pop[idy][0])
                    # if distance_sequencing > self.radius:
                    #     continue

                    # distance_sequencing = self.phenotypic_characristics[0].distance(phenotypic_characristics_pop[idx][0], phenotypic_characristics_pop[idy][0])
                    # distance_routing = self.phenotypic_characristics[0].distance(phenotypic_characristics_pop[idx][1], phenotypic_characristics_pop[idy][1])
                    # if distance_sequencing > self.radius and distance_routing > self.radius:
                    #     continue

                    if numWinners < self.capacity:
                        numWinners = numWinners + 1
                    else:
                        isCleared_pop[idy] = True
                        len_fitness_values = len(sorted_pop[idy].fitness.values)
                        bad_fitness = [np.Infinity for i in range(len_fitness_values)]
                        sorted_pop[idy].fitness.values = bad_fitness
                        clearedInds = clearedInds + 1

            print("Cleared number by niching: " + str(clearedInds))
        else:
            clearedInds = 0
            phenotypic_characristics_pop = []
            sorted_pop = self.sortPopulation(toolbox, population)
            isCleared_pop = []
            # calculate the PC of all individuals in population
            for idx in range(len(sorted_pop)):
                ind = sorted_pop[idx]
                sequencing_charList = self.phenotypic_characristics[0].characterise(ind[0])
                all_charList = []
                for char in sequencing_charList:
                    all_charList.append(char)
                phenotypic_characristics_pop.append(all_charList)
                isCleared_pop.append(False)

            # clear this population
            for idx in range(len(sorted_pop)):
                if isCleared_pop[idx]:
                    continue

                numWinners = 1
                for idy in range(idx + 1, len(sorted_pop)):
                    if isCleared_pop[idy]:
                        continue

                    distance = self.phenotypic_characristics[0].distance(phenotypic_characristics_pop[idx],
                                                                         phenotypic_characristics_pop[idy])
                    if distance > self.radius:
                        continue

                    if numWinners < self.capacity:
                        numWinners = numWinners + 1
                    else:
                        isCleared_pop[idy] = True
                        len_fitness_values = len(sorted_pop[idy].fitness.values)
                        bad_fitness = [np.Infinity for i in range(len_fitness_values)]
                        sorted_pop[idy].fitness.values = bad_fitness
                        clearedInds = clearedInds + 1

            print("Cleared number by niching: " + str(clearedInds))
        return sorted_pop


    def sortPopulation(self, toolbox, population):
        populationCopy = [toolbox.clone(ind) for ind in population]
        popsize = len(population)

        for j in range(popsize):
            sign = False
            for i in range(popsize - 1 - j):
                sum_fit_i = np.sum(populationCopy[i].fitness.values)
                sum_fit_i_1 = np.sum(populationCopy[i + 1].fitness.values)
                if sum_fit_i > sum_fit_i_1:
                    populationCopy[i], populationCopy[i + 1] = populationCopy[i + 1], populationCopy[i]
                    sign = True
            if not sign:
                break

        # FOR CHECK
        # pop_fit = [np.sum(ind.fitness.values) for ind in
        #            populationCopy]
        # print(pop_fit)
        return populationCopy