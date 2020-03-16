# -*- coding: utf-8 -*-
# Besiyata Dishmaya  בס"ד
#
# infer.py: do the inference in the EHR Challenge project
#
# ARIELize - [A]t [R]egular [I]ntervals [E]stimate [L]ongitunal data
# Copyright (C) 2018-2020 Ariel Yehuda Israel, M.D. Ph.D.
# Any utilization of this code/algorithm must credit the author
#
# ARIELize Electronic Health Records data from OMOP format
# then uses xgboost to build model and make predictions
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will help medical
# research, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# The terms of the GNU General Public License are available here:
# <https://www.gnu.org/licenses/>.

from arielize_ehr import do_infer

def main():
    do_infer()
    
                    
if __name__ == "__main__":
    main()
