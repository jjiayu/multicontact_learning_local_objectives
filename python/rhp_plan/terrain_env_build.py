#Scripts Building the environment model for optimization as well as the gazebo world file for simulation

# Generate terrain model (for python)


# save the terrain model

# construct the gazebo world file

with open('uneven_terrain.world', 'x') as f:
    f.write('<?xml version="1.0" ?>\n')
    f.write('<sdf version="1.4">\n')
    f.write('  <world name="default">\n')
    f.write('    <physics type="ode">\n')
    f.write('      <gravity>0 0 -9.81</gravity>\n')
    f.write('      <ode>\n')
    f.write('        <solver>\n')
    f.write('          <type>quick</type>\n')
    f.write('          <iters>50</iters>\n')
    f.write('          <sor>1.4</sor>\n')
    f.write('        </solver>\n')
    f.write('        <constraints>\n')
    f.write('          <cfm>0.0</cfm>\n')
    f.write('          <erp>0.2</erp>\n')
    f.write('          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>\n')
    f.write('          <contact_surface_layer>0.0</contact_surface_layer>\n')
    f.write('        </constraints>\n')
    f.write('      </ode>\n')
    f.write('      <real_time_update_rate>1000</real_time_update_rate>\n')
    f.write('      <max_step_size>0.001</max_step_size>\n')
    f.write('    </physics>\n')
    f.write('    <!-- A global light source -->\n')
    f.write('    <include>\n')
    f.write('      <uri>model://sun</uri>\n')
    f.write('    </include>\n')
    f.write('    <!-- A ground plane -->\n')
    f.write('    <include>\n')
    f.write('      <uri>model://ground_plane</uri>\n')
    f.write('    </include>\n')
    f.write('    <!-- A bunch of chairs -->\n')
    f.write('    <include>\n')
    f.write('          <uri>model://Y_positive_10_lifted</uri>\n')
    f.write('      <pose>1.0 0 0 0 0 0.0</pose>\n')
    f.write('    </include>\n')
    f.write('  </world>\n')
    f.write('</sdf>\n')

