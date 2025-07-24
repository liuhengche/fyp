class Edge(): 
    def __init__(self, f, t, id, length):
        self.f      = f
        self.t      = t
        self.id     = id
        self.length = length # unit: km

    def get_from_to(self): 
        return self.f, self.t
    
    def get_id(self): 
        return self.id
    
    def get_len(self): 
        return self.length
    
class Route(): 
    def __init__(self, o, d, links): 
        self.o = o
        self.d = d
        self.links = [link for link in links if link[0] != ':'] # list of link ids
        self.lengths = [0 for _ in self.links] # prefix sum (unit: km)

    def load_link_len(self, edges): 
        length = 0
        for idx, link in enumerate(self.links): 
            self.lengths[idx] = length
            length += edges[link].get_len()
        assert len(self.links) == len(self.lengths)

    def get_od_pair(self): 
        return (self.o, self.d)
    
    def get_lengths(self):
        assert self.lengths[-1] > 0, 'link length not loaded!'
        return self.lengths

    def get_route_len(self): 
        assert self.lengths[-1] > 0, 'link length not loaded!'
        return self.lengths[-1]

    def get_num_links(self): 
        assert len(self.links) == len(self.lengths)
        return len(self.links)

    def get_links(self):
        return self.links
    
    def get_last_link(self): 
        return self.links[-1]