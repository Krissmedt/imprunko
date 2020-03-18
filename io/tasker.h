#pragma once

#include <string>

#include "writers/writer.h"
#include "readers/reader.h"


namespace vlv{

template<size_t D>
inline void write_mesh( 
    corgi::Grid<D>& grid, 
    int lap,
    const std::string& dir 
    )
{

  std::string prefix = dir + "meshes-"; 
  prefix += std::to_string(grid.comm.rank());
  h5io::Writer writer(prefix, lap);

  for(auto cid : grid.get_local_tiles() ){
    const auto& tile 
      = dynamic_cast<vlv::Tile<D>&>(grid.get_tile( cid ));
    writer.write(tile);
  }
}


template<size_t D>
inline void read_mesh( 
    corgi::Grid<D>& grid, 
    int lap,
    const std::string& dir 
    )
{
  h5io::Reader reader(dir, lap, grid.comm.rank());

  for(auto cid : grid.get_tile_ids() ){
    auto& tile 
      = dynamic_cast<vlv::Tile<D>&>(grid.get_tile( cid ));
    reader.read(tile);
  }
}


}// end of namespace vlv


//--------------------------------------------------

namespace fields {

template<size_t D>
inline void write_yee( 
    corgi::Grid<D>& grid, 
    int lap,
    const std::string& dir
    )
{

  std::string prefix = dir + "fields-"; 
  prefix += std::to_string(grid.comm.rank());
  h5io::Writer writer(prefix, lap);

  for(auto cid : grid.get_local_tiles() ){
    const auto& tile 
      = dynamic_cast<fields::Tile<D>&>(grid.get_tile( cid ));
    writer.write(tile);
  }
}


//template<size_t D>
//inline void write_analysis( 
//    corgi::Grid<D>& grid, 
//    int lap,
//    const std::string& dir
//    )
//{
//
//  std::string prefix = dir + "analysis-"; 
//  prefix += std::to_string(grid.comm.rank());
//  h5io::Writer writer(prefix, lap);
//
//  for(auto cid : grid.get_local_tiles() ){
//    const auto& tile 
//      = dynamic_cast<fields::Tile<D>&>(grid.get_tile( cid ));
//    writer.write2(tile);
//  }
//}

template<size_t D>
inline void read_yee( 
    corgi::Grid<D>& grid, 
    int lap,
    const std::string& dir 
    )
{

  h5io::Reader reader(dir, lap, grid.comm.rank());

  for(auto cid : grid.get_tile_ids() ){
    auto& tile 
      = dynamic_cast<fields::Tile<D>&>(grid.get_tile( cid ));
    reader.read(tile);
  }

}

} // end of ns fields


//--------------------------------------------------

namespace pic {


template<size_t D>
void write_particles( 
    corgi::Grid<D>& grid, 
    int lap,
    const std::string& dir 
    )
{

  std::string prefix = dir + "particles-"; 
  prefix += std::to_string(grid.comm.rank());
  h5io::Writer writer(prefix, lap);

  for(auto cid : grid.get_local_tiles() ){
    const auto& tile 
      = dynamic_cast<pic::Tile<D>&>(grid.get_tile( cid ));
    writer.write(tile);
  }
}


template<size_t D>
inline void read_particles( 
    corgi::Grid<D>& grid, 
    int lap,
    const std::string& dir 
    )
{
  h5io::Reader reader(dir, lap, grid.comm.rank());

  for(auto cid : grid.get_tile_ids() ){
    auto& tile 
      = dynamic_cast<pic::Tile<D>&>(grid.get_tile( cid ));
    reader.read(tile);
  }
}


}
