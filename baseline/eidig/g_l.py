class EIDIG:
    def global_generation(self, X, seeds, decay=0.5, s_g=1.0):
        # global generation phase of EIDIG

        g_id = np.empty(shape=(0, self.num_attribs))
        all_gen_g = np.empty(shape=(0, self.num_attribs))
        try_times = 0
        g_num = len(seeds)
        for i in range(g_num):
            x1 = seeds[i].copy()
            grad1 = np.zeros_like(X[0]).astype(float)
            grad2 = np.zeros_like(X[0]).astype(float)
            for _ in range(self.max_iter):
                try_times += 1
                similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, self.model):
                    g_id = np.append(g_id, [x1], axis=0)
                    break
                x2 = generation_utilities.max_diff(x1, similar_x1, self.model)
                grad1 = decay * grad1 + self.compute_grad(x1)
                grad2 = decay * grad2 + self.compute_grad(x2)
                direction = np.zeros_like(X[0])
                sign_grad1 = np.sign(grad1)
                sign_grad2 = np.sign(grad2)
                for attrib in range(self.num_attribs):
                    if attrib not in self.protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                        direction[attrib] = (-1) * sign_grad1[attrib]
                x1 = x1 + s_g * direction
                x1 = generation_utilities.clip(x1, self.constraint)
                all_gen_g = np.append(all_gen_g, [x1], axis=0)
        g_id = np.array(list(set([tuple(id) for id in g_id])))
        return g_id, all_gen_g, try_times
    
    def local_generation(self, g_id, update_interval=5, s_l=1.0, epsilon=1e-6):
        # local generation phase of EIDIG

        direction = [-1, 1]
        l_id = np.empty(shape=(0, self.num_attribs))
        all_gen_l = np.empty(shape=(0, self.num_attribs))
        try_times = 0
        for x1 in g_id:
            x0 = x1.copy()
            similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
            x2 = generation_utilities.max_diff(x1, similar_x1, self.model)
            grad1 = self.compute_grad(x1)
            grad2 = self.compute_grad(x2)
            p = generation_utilities.normalization(grad1, grad2, self.protected_attribs, epsilon)
            p0 = p.copy()
            suc_iter = 0
            for _ in range(self.max_local):
                try_times += 1
                if suc_iter >= update_interval:
                    similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                    x2 = generation_utilities.find_pair(x1, similar_x1, self.model)
                    grad1 = self.compute_grad(x1)
                    grad2 = self.compute_grad(x2)
                    p = generation_utilities.normalization(grad1, grad2, self.protected_attribs, epsilon)
                    suc_iter = 0
                suc_iter += 1
                a = generation_utilities.random_pick(p)
                s = generation_utilities.random_pick([0.5, 0.5])
                x1[a] = x1[a] + direction[s] * s_l
                x1 = generation_utilities.clip(x1, self.constraint)
                all_gen_l = np.append(all_gen_l, [x1], axis=0)
                similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                if generation_utilities.is_discriminatory(x1, similar_x1, self.model):
                    l_id = np.append(l_id, [x1], axis=0)
                else:
                    x1 = x0.copy()
                    p = p0.copy()
                    suc_iter = 0
        l_id = np.array(list(set([tuple(id) for id in l_id])))
        return l_id, all_gen_l, try_times