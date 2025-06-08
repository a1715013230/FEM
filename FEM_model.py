# -*- coding: utf-8 -*-
import numpy as np
import random

# from design_parameters import *

from abaqus import *
from abaqusConstants import *
import sketch  
import sketchPlane
import part
import assembly
import material
import step
import interaction
import regionToolset
from mesh import *
import time
import visualization
import odbAccess
import math
import datetime
# work path

## Read the voxel quantities in three directions and their material distribution in 'f'
## with ensures the file is automatically closed after operations

def read_npy(file_path):
    """
    Read multiple material distributions from an .npy file.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        list: A list of tuples, each containing (nx, ny, nz, locations).
    """
    distributions = []  # List to store all distributions

    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)

    for distribution in data:
        nx, ny = distribution.shape
        distributions.append((nx, ny, distribution))

    return distributions


## The class is used to define the creation of a model in Abaqus

class Active2DStructure:

## Define a function to receive initial parameters
## __init__ is used for initialization operations and is automatically called when creating an instance of the class
    def __init__(self,
                 length_x, length_y,  nx, ny,
                 voxel_size_x, voxel_size_y,
                 locations, num_cpus,
                 alpha, temp_low, temp_high,
                 seed_x, seed_y,
                 stepTime, iniInc, maxInc):
        """
        Class constructor
        """
         ## Record the initial time

        initial = datetime.datetime.now()   ## Module.Class.Method
        print('Started at ' + str(initial))

        ## Initialize attributes
        self.num_cpus = num_cpus

        self.length_x = length_x
        self.length_y = length_y

        self.nx = nx
        self.ny = ny
        self.voxel_size_x = voxel_size_x
        self.voxel_size_y = voxel_size_y

        self.locations = locations # voxel info

        self.alpha = alpha
        self.temp_low = temp_low
        self.temp_high = temp_high

        self.seed_x = seed_x
        self.seed_y = seed_y

        self.stepTime = stepTime
        self.iniInc = iniInc
        self.maxInc = maxInc
        ## Create Abaqus model
        self.model_name = 'Candidate' + str(random.randint(0, 500000))
        self.m = mdb.Model(name=self.model_name)## Use mdb module to create a new model instance and assign it to self.m
        #
        # start calling all functions
        #
        ## Call class methods
        self.generate_geometry()
        self.generate_assembly()
        self.create_and_assign_materials_and_sections()
        # self.total_number_of_chunks = self.create_assembly_instance_sets()
        self.create_steps()
        # self.initialize_interactions()
        self.set_initial_conditions_and_step_temperatures()
        self.set_boundary_conditions()
        self.mesh_structure()

        self.m.rootAssembly.regenerate()

        print('Started at ' + str(initial))
        print('Completed at ' + str(datetime.datetime.now()) )

    def generate_geometry(self):
        """
        Generates the geometry of a single voxel by creating a rectangular sketch
        and extruding it to form a solid part.
        """
        print("Creating geometry: voxel_size_x=" + str(self.voxel_size_x) + " mm, voxel_size_y=" + str(self.voxel_size_y) + " mm")
        
        # Ensure dimensions are valid
        MIN_SIZE = 0.0001  # Minimum size limit
        if self.voxel_size_x < MIN_SIZE or self.voxel_size_y < MIN_SIZE:
            print("Warning: Voxel size too small, adjusting to minimum "+str(MIN_SIZE)+" mm")
            
        voxel_size_x = max(self.voxel_size_x, MIN_SIZE)
        voxel_size_y = max(self.voxel_size_y, MIN_SIZE)

        try:
            # Create constrained sketch
            sketch_prof = self.m.ConstrainedSketch(name='Profile-Sketch',
                                                 sheetSize=max(20.0, 2*voxel_size_x, 2*voxel_size_y))
            
            # Draw rectangle
            sketch_prof.rectangle(point1=(0.0, 0.0),
                                 point2=(voxel_size_x, voxel_size_y))
            
            # Create part
            part_unit = self.m.Part(name='Part-Voxel',
                                    dimensionality=TWO_D_PLANAR,
                                    type=DEFORMABLE_BODY)
            
            # Create shell from sketch
            part_unit.BaseShell(sketch=sketch_prof)
            print("Geometry created successfully")
        except Exception as e:
            print("Failed to create geometry: "+str(e))
            raise

    def generate_assembly(self):
        """
        Assemble the created parts by translating them to the corresponding positions
        """
        
        a = self.m.rootAssembly  # Get the root assembly object
        a.DatumCsysByDefault(CARTESIAN)  # Ensure using Cartesian coordinate system

        voxels_Inst = []
        # Create and translate voxel instances
        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                instance_name = 'Assem-voxel-xi_' + str(ix) + '-yi_' + str(iy)
                try:
                    a.Instance(name=instance_name,
                              part=self.m.parts['Part-Voxel'],
                              dependent=ON)
                    # Create a new part instance and add to assembly
                    a.translate(instanceList=[instance_name],
                               vector=(ix*self.voxel_size_x, iy*self.voxel_size_y, 0.0))
                    voxels_Inst.append(a.instances[instance_name])
                except Exception as e:
                    print("Error creating instance "+instance_name+": "+str(e))

        if not voxels_Inst:
            raise Exception("No voxel instances were successfully created")


        # Merge all voxel instances into one part
        newName = 'Part-All'
        try:
            a.InstanceFromBooleanMerge(instances=voxels_Inst,  # Get instance list
                                      keepIntersections=ON,    # Retain information on intersecting regions
                                      domain=GEOMETRY,         # Operation scope for the entire geometry
                                      mergeNodes=NONE,         # Do not merge nodes
                                      name=newName,            # Name
                                      originalInstances=SUPPRESS)  # Suppress original instances after merging
            print("Voxel merging successful")
        except Exception as e:
            print("Error while merging voxels: "+str(e))
            raise

    def create_and_assign_materials_and_sections(self):
        """
        Create materials and assign sections
        """
        ## Create active material
        materActive = self.m.Material(name='Active_Material')
        materActive.Hyperelastic(type=NEO_HOOKE, testData=OFF, table=((0.1, 0.002,),))   ## Material model neo_hooke, non-linear elastic model
        materActive.Expansion(type=ISOTROPIC, table=((self.alpha,),))   ## Isotropic expansion
        materActive.Density(table=((1e-6,),))

        materPassive = self.m.Material(name='Passive_Material')
        materPassive.Hyperelastic(type=NEO_HOOKE, testData=OFF, table=((0.1, 0.002,),))
        materPassive.Expansion(type=ISOTROPIC, table=((0.0,),))
        materPassive.Density(table=((1e-6,),))

        ## Create section types and associate them
        # For 2D plane model, use plane strain section
        self.m.HomogeneousSolidSection(material='Active_Material',
                                       name='Active_Section',
                                       thickness=1.0)

        self.m.HomogeneousSolidSection(material='Passive_Material',
                                       name='Passive_Section',
                                       thickness=1.0)

        ## Set up parts
        p = self.m.parts['Part-All']

        # Create face set instead of cell set for 2D part
        p.Set(name='All', faces=p.faces[:])

        ## Create sets for each voxel
        # ======= MATERIAL ASSIGNMENT SECTION ======= #
        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                    pSetName = 'Section-voxel-xi_' + str(ix) + '-yi_' + str(iy)
                    x1 = ix * self.voxel_size_x
                    y1 = iy * self.voxel_size_y
                    x2 = (ix+1) * self.voxel_size_x
                    y2 = (iy+1) * self.voxel_size_y

                    # Get faces within this region
                    face_temp = p.faces.getByBoundingBox(x1, y1, -0.001, x2, y2, 0.001)
                    if not face_temp:
                        print("Warning: No faces found in region ("+str(x1)+","+str(y1)+") to ("+str(x2)+","+str(y2)+")")
                        continue
                        
                    p.Set(name=pSetName, faces=face_temp)

        ## Assign material properties to each voxel
        active_face = ()
        passive_face = ()
        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                pSetName = 'Section-voxel-xi_' + str(ix) + '-yi_' + str(iy)
                if pSetName not in p.sets:
                    continue
                
                s = p.sets[pSetName]
                if self.locations[ix, iy] == 1:
                    active_face = active_face + (s.faces,)
                elif self.locations[ix, iy] == 0:   
                    passive_face = passive_face + (s.faces,)

        # Create face sets for active and passive materials
        if active_face:
            active_set = p.Set(name='pset_Active', faces=active_face)  # Part set for all active regions
        if passive_face:
            passive_set = p.Set(name='pset_Passive', faces=passive_face)  # Part set for all passive regions

        ## Assign sections
            p.SectionAssignment(region=p.sets['pset_Active'],
                               sectionName='Active_Section')
            p.SectionAssignment(region=p.sets['pset_Passive'],
                               sectionName='Passive_Section')

    def create_steps(self):
        """
        Create analysis steps, including viscoelastic analysis steps and field output request configuration.

        Steps:
        1. Create a viscoelastic analysis step (ViscoStep).
        2. Configure the time interval for the field output request.
        3. (Optional) Configure control parameters for the analysis step, such as time increments and field settings.
        """
        self.m.ViscoStep(name='Actuation',
                        previous='Initial',        ## Preceding step name, indicating that this step follows the 'Initial' step
                        timePeriod=self.stepTime,  ## Initial time increment, indicating the time step at the start of the analysis
                        initialInc=self.iniInc,
                        minInc=self.stepTime * 1.e-8,
                        maxInc=self.maxInc,
                        nlgeom=ON,                 ## Enable geometric nonlinearity to consider large deformation effects
                        cetol=0.05,                ## Convergence tolerance to control the convergence standard of the iteration process
                        maxNumInc=100)

        ## Set the time interval for the field output request 'F-Output-1' to 10
        self.m.fieldOutputRequests['F-Output-1'].setValues(numIntervals=10)
        """
        self.m.steps['Actuation'].control.setValues(allowPropagation=OFF, 
            discontinuous=ON, resetDefaultValues=OFF, timeIncrementation=(8.0, 10.0, 
            9.0, 16.0, 10.0, 4.0, 12.0, 20.0, 6.0, 3.0, 50.0))
        
        self.m.steps['Actuation'].control.setValues(allowPropagation=
            OFF, displacementField=(0.5, 1.0, 0.0, 0.0, 0.02, 1e-05, 0.001, 1e-08, 1.0, 
            1e-05, 1e-08), hydrostaticFluidPressureField=DEFAULT, resetDefaultValues=
            OFF, rotationField=DEFAULT, temperatureField=DEFAULT)
        """

    def set_boundary_conditions(self):
            #
            a = self.m.rootAssembly

            myIns = a.instances['Part-All-1']

            # === create set for faces ===
            #
    ## getByBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax): Defines a bounding box in three-dimensional space
    ## selects all faces located within the bounding box.

            edge_left = myIns.edges.getByBoundingBox(-0.001, -0.001, -0.001,
                                                    0.001, 0.001 + self.length_y, 0.001)
            setEdgeLeft = a.Set(name='set-EdgeLeft', edges = edge_left)
            
            edge_mid = myIns.edges.getByBoundingBox(-0.001, self.length_y/2-0.01, -0.001,
                                        self.length_x+0.001, self.length_y/2+0.01, 0.001)
            setEdgeMid = a.Set(name='SET-EDGEMID', edges = edge_mid)

            # ==== Initial BCs ====
            self.m.DisplacementBC(name='BC-1-edge_left',
                                 createStepName='Initial',
                                 region=a.sets['set-EdgeLeft'],
                                 u1=0,u2=0, u3=0)

    def set_initial_conditions_and_step_temperatures(self):
        """
        Set initial temperature conditions and control the temperature variation over time
        """
        myIns = self.m.rootAssembly.instances['Part-All-1']
        # Note: myIns.sets have the sets defined in Part.
        #       self.m.rootAssembly.sets[''] only have the sets defined in Assembly.

        # Check if set exists
        if 'All' not in myIns.sets:
            print("Warning: Set 'All' does not exist, trying to create")
            # Create a set containing all faces for the instance
            all_faces = myIns.faces[:]
            if all_faces:
                self.m.rootAssembly.Set(name='All', faces=all_faces)
            else:
                print("Error: Cannot find faces")
                return

        try:
            temp_field = self.m.Temperature(name='Initial-Temp',
                                   crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
                                   distributionType=UNIFORM,
                                   createStepName='Initial',
                                   region=myIns.sets['All'],
                                   magnitudes=self.temp_low)
        except Exception as e:
            print("Error setting temperature field: "+str(e))
            # Try alternative method
            try:
                temp_field = self.m.Temperature(name='Initial-Temp',
                                       distributionType=UNIFORM,
                                       createStepName='Initial',
                                       region=self.m.rootAssembly.instances['Part-All-1'],
                                       magnitudes=self.temp_low)
            except Exception as e:
                print("Alternative temperature setting method also failed: "+str(e))
                return

        # Define temperature variation over time
        self.m.TabularAmplitude(name='heating', timeSpan=STEP,
                        data=((0.0, self.temp_low/self.temp_high),
                              (self.stepTime, 1.0),))

        # Apply temperature change during analysis step
        try:
            temp_field.setValuesInStep(stepName='Actuation',
                                      magnitudes=self.temp_high, amplitude='heating')
        except Exception as e:
            print("Error setting step temperature: "+str(e))

    def mesh_structure(self):
        """
        Mesh the structure with appropriate element types and seeds
        """
        part = self.m.parts['Part-All']
        
        # For 2D planar model, use faces
        faces = part.faces[:]
        
        # Set mesh controls
        part.setMeshControls(regions=faces,
                             elemShape=QUAD)
        part.setElementType(elemTypes=(ElemType(elemCode=CPE4H, elemLibrary=STANDARD),),
                           regions=part.sets['All'])
                                      
        
        # Set global seed
        global_seed = self.seed_x
        part.seedPart(size=global_seed)
        
        # Use specific seed for vertical edges (may need to adjust edge selection method)
        try:
            # Try using findAt method
            edge_vertical = part.edges.findAt(((0.0, 0.5*self.length_y, 0.0),))
            part.seedEdgeBySize(edges=edge_vertical, size=self.seed_y, constraint=FIXED)
        except:
            # If findAt fails, use getByBoundingBox as fallback
            edge_vertical = part.edges.getByBoundingBox(-0.001, -0.001, -0.001,
                                                       0.001, self.length_y+0.001, 0.001)
            if edge_vertical:
                part.seedEdgeBySize(edges=edge_vertical, size=self.seed_y, constraint=FIXED)
        
        # Generate mesh
        part.generateMesh()


    def run_simulation(self, job_name, save_CAE):

        if save_CAE == True:
            mdb.saveAs(pathName=job_name[0:5]+'CAE')
            #return True

        mdb.Job(contactPrint=OFF, description='', echoPrint=OFF,
                explicitPrecision=SINGLE, historyPrint=OFF,
                memory=90, memoryUnits=PERCENTAGE, model=self.model_name,
                modelPrint=OFF, multiprocessingMode=DEFAULT, name=job_name,
                nodalOutputPrecision=SINGLE, numCpus=self.num_cpus, numDomains=self.num_cpus,
                parallelizationMethodExplicit=DOMAIN, scratch='', type=ANALYSIS,
                userSubroutine='')
        mdb.jobs[job_name].submit()
        mdb.jobs[job_name].waitForCompletion()

        # Below is used to check the job.
        with open(job_name+'.log','r') as f_log:
            f_log.seek(-10,2)
            status_log = f_log.read(9)
        if status_log == 'COMPLETED':
            aborted = False
        else:
            aborted = True

        return aborted

    @staticmethod
    def csv_output(aborted, job_name, output_dir):
        """
        Extract displacement data of specific nodes from the Abaqus ODB file and export it to a CSV file.

        Parameters:
            aborted (bool): Indicates whether the simulation was aborted.
            job_name (str): The name of the simulation job, used to locate the relevant ODB file.
            output_dir (str): The directory where the CSV file will be saved.

        Returns:
            None
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Construct the output file path
        output_path = os.path.join(output_dir, "outs-" + job_name + ".csv")

        if aborted:
            print('*Simulation not completed for error_code_1')
            return

        # Try to open the ODB file
        try:
            odb = session.openOdb(name=job_name + '.odb')
        except RuntimeError as e:
            print("Error opening ODB file: " + str(e))
            return

        if len(odb.steps['Actuation'].frames) == 0:
            print("No frames found in 'Actuation' step.")
            return

        # Extract node displacement data
        nodes = odb.rootAssembly.nodeSets['SET-EDGEMID']
        node_labels, x_coords, y_coords  = [], [], []
        for node in nodes.nodes[0]:
            node_labels.append(node.label)
            x_coords.append(node.coordinates[0])
            y_coords.append(node.coordinates[1])

        x_disp, y_disp, z_disp = [], [], []
        for i_f, frame in enumerate(odb.steps['Actuation'].frames):
            if i_f == 0:
                continue
            u = frame.fieldOutputs['U'].getSubset(region=nodes)
            x_disp_temp, y_disp_temp, z_disp_temp = [], [], []
            for v in u.values:
                x_disp_temp.append(v.data[0])
                y_disp_temp.append(v.data[1])

            x_disp.append(x_disp_temp)
            y_disp.append(y_disp_temp)

        # Write data to CSV
        # Construct CSV file headers, including node labels, coordinates, and displacement data fields for each frame
        fieldnames = ['node_label', 'x_coord', 'y_coord']
        for i_f in range(1, len(odb.steps['Actuation'].frames)):
            fieldnames.append('f' + str(i_f) + '_x_disp')
            fieldnames.append('f' + str(i_f) + '_y_disp')


        with open(output_path, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(node_labels)):
                row = {
                    'node_label': node_labels[i],
                    'x_coord': x_coords[i],
                    'y_coord': y_coords[i],
                }
                for i_f in range(len(x_disp)):
                    row['f' + str(i_f+1) + '_x_disp'] = x_disp[i_f][i]
                    row['f' + str(i_f+1) + '_y_disp'] = y_disp[i_f][i]
                writer.writerow(row)


        # Close the ODB file
        try:
            odb.close()
        except RuntimeError:
            pass


def clear_directory(directory='.'):
        """
        Clear unnecessary files in the specified directory.

        Args:
            directory (str): The directory to clear. Defaults to the current directory.
        """
        # Extensions of files to be removed
        extensions = ['png', 'rec', 'sta', 'msg', 'rpy',
                    'dat', 'log', 'inp', 'com', 'prt',
                    'sim', 'ipm', 'mdl', 'stt', '1', '023', 'csv']

        # List all files in the directory
        files = os.listdir(directory)

        for file in files:
            # Get the file extension
            file_path = os.path.join(directory, file)
            try:
                e = file.split(".")[1]
            except IndexError:
                e = ""

            # If the file has an unwanted extension, remove it
            if e in extensions:
                try:
                    os.remove(file_path)
                    print("Removed: " + file_path)
                except OSError as err:
                    print("Error removing " + file_path + ": " + str(err))

def main(npy_file, csv_output_dir, start_index=0):
    # Beam physical dimensions (mm)
    length_x, length_y = 80.0, 1.0
    
    # Total number of elements in the mesh
    total_elements_x = 960
    total_elements_y = 12
    
    # Other parameters
    active_strain = 0.1
    temp_low, temp_high = 0.0, 100.0
    alpha = active_strain / (temp_high - temp_low)
    stepTime = 5.0
    iniInc = stepTime / 5.0
    maxInc = stepTime / 2.0
    num_cpus = 4

    # Read material distributions from .npy file
    distributions = read_npy(npy_file)

    # Validate start_index
    if start_index < 0 or start_index >= len(distributions):
        raise ValueError("Invalid start_index " + str(start_index) +
                 ". It should be between 0 and " + str(len(distributions) - 1) + ".")


    # Iterate over distributions and run simulations from the specified start_index
    for idx, (nx, ny, locations) in enumerate(distributions[start_index:], start=start_index):

        

        voxel_size_x = float(length_x) / float(nx)
        voxel_size_y = float(length_y) / float(ny)
        
        

        elements_per_voxel_x = total_elements_x / nx
        elements_per_voxel_y = total_elements_y / ny
        
        seed_x = voxel_size_x / elements_per_voxel_x
        seed_y = voxel_size_y / elements_per_voxel_y
        

        job_name = "Job_Distribution_" + str(idx + 1)
        model = Active2DStructure(length_x, length_y, nx, ny,
                                  voxel_size_x, voxel_size_y,
                                  locations, num_cpus,
                                  alpha, temp_low, temp_high,
                                  seed_x, seed_y,
                                  stepTime, iniInc, maxInc)

        save_CAE = False
        aborted = model.run_simulation(job_name, save_CAE)
        clear_directory()

        model.csv_output(aborted, job_name, csv_output_dir)


if __name__ == '__main__':
    # Input and output paths
    npy_file_in = r"F:\Academy\shcolar\2D_beam\data\distributions\Distribution_300.npy"
    csv_output_dir = r"F:\Academy\shcolar\2D_beam\data\CSV\CSV_disp_300"
    start_index = 0
    main(npy_file_in, csv_output_dir, start_index=start_index)
