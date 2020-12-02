"""

"""

from view.view import View
from services.services import Service

import sys

class Controller(object):


    def __init__(self):
        self.view = View()
        self.services = Service()        


    def startSystem(self):
        while True:
            option = self.view.menu()
            self.operations(option)
            
    
    def operations(self, option):
        if option == 1:
            second_option = self.view.submenu()
            if second_option == 1:
                p_file = self.view.select_path()
                self.services.open_image(p_file)
            elif second_option == 2:
                p_file = self.view.select_path()
                xh, xw, yh, yw = self.view.get_parameters()               
                self.services.cut_image(p_file, xh, xw, yh, yw)
            elif second_option == 3:
                p_file = self.view.select_path()
                width, height = self.view.get_wh() 
                self.services.resize_image(p_file, width, height)
            elif second_option == 4:
                p_file = self.view.select_path()
                color_option = self.view.select_color_option()                
                if color_option == 4:
                    return self.startSystem()
                self.services.color_image(p_file, color_option)
            else:
                self.startSystem()

        elif option == 2:
            third_option = self.view.detecmenu()            
            if third_option == 1:
                p_file = self.view.select_path()
                detec_option = self.view.sel_algorithm()
                if detec_option == 3:
                    return self.startSystem()
                self.services.detect_person(p_file, detec_option)
            elif third_option == 2:
                p_file = self.view.select_path()
                detec_option = self.view.sel_algorithm()
                if detec_option == 3:
                    return self.startSystem()
                self.services.detect_person(p_file, detec_option)
            elif third_option == 3:          
                detec_option = self.view.sel_algorithm()      
                if detec_option == 3:
                    return self.startSystem()
                self.services.detect_person(None, detec_option)
            elif third_option == 4:
                p_file = self.view.select_path()
                c_object = self.view.object_color_menu()
                self.view.print_message2()
                self.services.get_object(p_file, c_object)
            elif third_option == 5:                
                c_object = self.view.object_color_menu()
                self.view.print_message2()
                self.services.get_object(None, c_object)
            else:
                self.startSystem()

        elif option == 3:
            fourth_option = self.view.morphmenu("FOTO")
            if fourth_option == 1:
                p_file = self.view.select_path()
                self.services.binarization(p_file)
            elif fourth_option == 2:
                p_file = self.view.select_path()
                n_iter = self.view.num_iterations()
                self.services.morphology(p_file, n_iter)
            elif fourth_option == 3:
                p_file = self.view.select_path()
                self.view.print_message()
                self.services.hsv(p_file, option)
            else:
                self.startSystem()

        elif option == 4:
            fifth_option = self.view.morphmenu("VIDEO")
            if fifth_option == 1:
                p_video = self.view.select_path()
                self.services.binarization(p_video)
            elif fifth_option == 2:
                p_video = self.view.select_path()
                n_iter = self.view.num_iterations()
                self.services.morphology(p_video, n_iter)
            elif fifth_option == 3:
                p_video = self.view.select_path()
                self.view.print_message()
                self.services.hsv(p_video, option)
            else:
                self.startSystem()

        else:
            sys.exit()
