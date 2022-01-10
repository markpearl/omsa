//------------------------1. PREPARATION------------------------//
//-----------------------------SVG------------------------------//
const margin = 250
const global_height = 700 
const global_width = 1200
const width = global_width - margin
const height = global_height - margin

//Appending dropdown to body of html page 
d3.select('#gameDropdown')
    .attr("style","max-width:90%;")

//Appending svg to body
var svg = d3.select("body")
    .append("svg")
    .attr("height", global_height)
    .attr("width", global_width);        


var path = d3.geoPath();

var projection = d3.geoNaturalEarth2()
    .center([30,30])
    .scale(height / 2)
    .translate([width-200,300]);        
    
// Data and color scale
var data = d3.map();


Promise.all(
     [
    d3.json('world_countries.json'),
      d3.csv('ratings-by-country.csv')]).
      then(
          ([world, gameData]) => 
          {
            var gameDataCleansed = gameData.map(function(d,i){
                return {
                    game: d["Game"],
                    country: d["Country"],
                    average_rating: parseFloat(d["Average Rating"]),
                    num_users_rated : parseInt(d["Number of Users"])
                }
            })
            var filtered_world = world.features.filter(function(d) { return d.id != 'ATA'})

            // enter code to extract all unique games from gameData
            console.log(world)
            // enter code to append the game options to the dropdown
            var unique = [...new Set(gameDataCleansed.map(item => item.game))];
            // add the options to the button
            d3.select("#gameDropdown")
                .selectAll("options")
                .data(unique.sort())
                .enter()
                .append('option')
                .text(function (d) { return d; }) // text showed in the menu
                .attr("value", function (d) { return d; })     
            defaultGame = unique[0];
            createMapAndLegend(filtered_world,gameDataCleansed,defaultGame)

            d3.select("#gameDropdown").on("click", 
                function(d) {
                // create Choropleth with default option. Call createMapAndLegend() with required arguments. 
                selectedGame = d3.select('#gameDropdown option:checked').text();
                createMapAndLegend(filtered_world,gameDataCleansed,selectedGame)
            })
// this function should create a Choropleth and legend using the world and gameData arguments for a selectedGame
// also use this function to update Choropleth and legend when a different game is selected from the dropdown
function createMapAndLegend(world, gameData, selectedGame){ 
    //Add average rating to worlds data
    for (let i = 0; i < world.length; i++) {
        var current_country = world[i].properties.name
        var average_rating_country = gameData.filter(function(d) { return d.country  == current_country && d.game==selectedGame})
        var average_rating = average_rating_country.length ==0 ? 0: average_rating_country[0].average_rating; 
        world[i]['average_rating']=average_rating
    }    
    
    colorScale = d3.scaleQuantile().domain([d3.min(gameData.filter(function(d) { return d.game==selectedGame}),function(c) { return c.average_rating;}),
    d3.max(gameData.filter(function(d) { return d.game==selectedGame}),function(c) { return c.average_rating;})]).range(d3.schemeBlues[7]);
    


    const countryRegions = svg.append("g")
    .attr("id","countries")    
    .attr("width",global_width)
    .attr("height",global_height)
    
    // Draw the map
    countryRegions
    .selectAll("path")
    .data(world)
    .enter()
    .append("path")
        // draw each country
        .attr("d", d3.geoPath()
        .projection(projection)
        )
    .attr("fill", 
    function(d) { 
        return d.average_rating == 0 ? "grey" : colorScale(d.average_rating)})



}});