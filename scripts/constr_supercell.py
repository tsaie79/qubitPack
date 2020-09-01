import os
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Poscar
#from shutil import copyfile

root_1 = [[1,0,0],[0,1,0],[0,0,1]]
root_3 = [[1,1,0],[-1,2,0],[0,0,1]]
root_4 = [[2,0,0],[0,2,0],[0,0,1]]
root_7 = [[2,1,0],[-1,3,0],[0,0,1]]
root_9 = [[3,0,0],[0,3,0],[0,0,1]]
root_12 = [[2,2,0],[-2,4,0],[0,0,1]]
root_13 = [[3,1,0],[-1,4,0],[0,0,1]]
root_16 = [[4,0,0],[0,4,0],[0,0,1]]

input = open('smallest_supercell_strain_lt_5percent.dat','r')
for line in input:
	#020289_Bi2Te3 4.350 root_3 042555_Pd1Te2 3.963 root_4 0.049
	word = line.split('\n')[0]
	str1 = word.split()[0]
	size1 = word.split()[2]
	str2 = word.split()[3]
	size2 = word.split()[5]	
	os.chdir(str1+'_'+str2)
	s1 = Structure.from_file(str1+'.vasp') #r13
	s2 = Structure.from_file(str2+'.vasp') #r12
	s1.lattice.angles[2]
	s2.lattice.angles[2]
	if abs(s1.lattice.angles[2] - 60.0) < abs(s1.lattice.angles[2] - 120.0):
    		s1.make_supercell([[1,0,0],[0,1,0],[0,0,1]])
	else:
	    s1.make_supercell([[1,0,0],[1,1,0],[0,0,1]])
	if abs(s2.lattice.angles[2] - 60.0) < abs(s2.lattice.angles[2] - 120.0):
	    s2.make_supercell([[1,0,0],[0,1,0],[0,0,1]])
	else:
	    s2.make_supercell([[1,0,0],[1,1,0],[0,0,1]])
	#for i in [root_1,root_3,root_4,root_7,root_9,root_12,root_13,root_16]:
	if size1 == 'root_1':
		s1.make_supercell(root_1)
	elif size1 == 'root_3':
		s1.make_supercell(root_3)
	elif size1 == 'root_4':
		s1.make_supercell(root_4)
	elif size1 == 'root_7':
		s1.make_supercell(root_7)
	elif size1 == 'root_9':
		s1.make_supercell(root_9)
	elif size1 == 'root_12':
		s1.make_supercell(root_12)
	elif size1 == 'root_13':
		s1.make_supercell(root_13)
	elif size1 == 'root_16':
		s1.make_supercell(root_16)
	Poscar(s1).write_file('POSCAR1')
	#for j in [root_1,root_3,root_4,root_7,root_9,root_12,root_13,root_16]:
	if size2 == 'root_1':
		s2.make_supercell(root_1)
	elif size2 == 'root_3':
		s2.make_supercell(root_3)
	elif size2 == 'root_4':
		s2.make_supercell(root_4)
	elif size2 == 'root_7':
		s2.make_supercell(root_7)
	elif size2 == 'root_9':
		s2.make_supercell(root_9)
	elif size2 == 'root_12':
		s2.make_supercell(root_12)
	elif size2 == 'root_13':
		s2.make_supercell(root_13)
	elif size2 == 'root_16':
		s2.make_supercell(root_16)
	Poscar(s2).write_file('POSCAR2')
	a1,b1,c1 = s1.lattice.abc
	a2,b2,c2 = s2.lattice.abc
	list1 = []
	list2 = []
	for i in s1.cart_coords:
		list1.append(i[2])
	height1 = max(list1)-min(list1)
	for j in s2.cart_coords:
		list2.append(j[2])
	height2 = max(list2)-min(list2)
	c = height1+height2+18.0
	pos1 = open('POSCAR1','r')	
	pos2 = open('POSCAR2','r')
	pos = open('POSCAR','w')
	f1 = pos1.readlines()
	f2 = pos2.readlines()
	pos.write('heter'+'\n')
	pos.write(f1[1])
	if a1 > a2:
		pos.write(f1[2])
		pos.write(f1[3])
	else:
		pos.write(f2[2])
		pos.write(f2[3])
	pos.write('   0.0000   0.00000   '+str(c)+'\n')
	pos.write(f1[5].split('\n')[0]+' '+f2[5].split('\n')[0]+'\n')
	pos.write(f1[6].split('\n')[0]+' '+f2[6].split('\n')[0]+'\n')
	pos.write('Direct'+'\n')
	shift1 = c/2.0+1.5 - min(list1)
	shift2 = c/2.0-1.5 - max(list2)
	for m in range(8,8+s1.num_sites): 
	    x1 = f1[m].split('\n')[0].split()[0]
	    y1 = f1[m].split('\n')[0].split()[1]
	    z1 = f1[m].split('\n')[0].split()[2]
	    z1 = (float(z1)*c1+shift1)/c
	    pos.write(str(x1)+'   '+str(y1)+'   '+str(z1)+'\n')
	for n in range(8,8+s2.num_sites):
	    x2 = f2[n].split('\n')[0].split()[0]
	    y2 = f2[n].split('\n')[0].split()[1]
	    z2 = f2[n].split('\n')[0].split()[2]
	    z2 = (float(z2)*c2+shift2)/c
	    #print(z2)
	    pos.write(str(x2)+'   '+str(y2)+'   '+str(z2)+'\n')
	pos1.close()
	pos2.close()
	pos.close()
	
	s=Structure.from_file('POSCAR')
	Poscar(s).write_file('POSCAR')
	os.chdir('../')
input.close()
