# -*- coding: utf-8 -*-
# Besiyata Dishmaya  בס"ד
#
# train.py: do the training in the EHR Challenge project
#
# ARIELize - [A]t [R]egular [I]ntervals [E]stimate [L]ongitunal data
# Copyright (C) 2018-2020 Ariel Yehuda Israel, M.D. Ph.D.
# Any utilization of this code/algorithm must credit the author
#
# ARIELize Electronic Health Records data from OMOP format
# then uses xgboost to build model and make predictions
#
# This program is distributed in the hope that it will help medical
# research, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# The terms of the GNU General Public License are available here:
# <https://www.gnu.org/licenses/>.


from arielize_ehr import do_train, list_files

def main():
    do_train()    
    
                    
if __name__ == "__main__":
    main()
