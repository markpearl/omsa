//------------------------1. PREPARATION------------------------//
//-----------------------------SVG------------------------------//
const margin = 250
const global_height = 500 
const global_width = 960
const width = global_width - margin
const height = global_height - margin
// we are appending SVG first

const svg_linechart = d3.select("body")
        .append("svg")
        .attr("width",global_width)
        .attr("height", global_height)

//-----------------------------DATA-----------------------------//
const timeConv = d3.timeParse("%Y-%m-%d");
const dataset = d3.dsv(",","average-rating.csv");
dataset.then(function(data) {

    
    //Create rawArr for creating the line data in chart a
    var rawArr = data.map(function(d,i) {
        return d.year+'|'+Math.floor(d.average_rating)
    });
    function countAndSort(arr) { 
        return Object.entries(arr.reduce((prev, curr) => (prev[curr] = ++prev[curr] || 1, prev), {})).sort((a,b) => b[1]-a[1])
    } ;
    var frequencyCount = countAndSort(rawArr).map(function(d,i){
        return {
            year: d[0].split('|')[0],
            average_rating: parseInt(d[0].split('|')[1]),
            frequency:  parseInt(d[1]),
        }
    })

    var ratingPerYear ={};
    frequencyCount.forEach(element => {
        var makeKey = element.year;
         if(!ratingPerYear[makeKey]) {
            ratingPerYear[makeKey] = [];
         }
    
         ratingPerYear[makeKey].push(element.average_rating);
       });
    
    function range(start, end) {
        return Array(end - start + 1).fill().map((_, idx) => start + idx)
    }
    var fullRatingsRange = range(0,9)       

    for (let i = 0; i < Object.keys(ratingPerYear).length; i++) {
        year = Object.keys(ratingPerYear)[i]
        if (year >= 2015 && year <= 2019) {
            currentRatings = ratingPerYear[year]
            missingRatings = fullRatingsRange.filter(x => !currentRatings.includes(x));
            missingRatings.map(function(d){
                missingDict = {
                    year: year,
                    average_rating: d,
                    frequency: 0
                }
                frequencyCount.push(missingDict)

            })
          }
        }

       
    console.log(ratingPerYear)
    function fieldSorter(fields) {
        return function (a, b) {
            return fields
                .map(function (o) {
                    var dir = 1;
                    if (o[0] === '-') {
                       dir = -1;
                       o=o.substring(1);
                    }
                    if (a[o] > b[o]) return dir;
                    if (a[o] < b[o]) return -(dir);
                    return 0;
                })
                .reduce(function firstNonZeroValue (p,n) {
                    return p ? p : n;
                }, 0);
        };
    }

    var frequencyCountsSorted = frequencyCount.sort(fieldSorter(['year','average_rating']))
    
    var frequencyGroupedYear ={};
    frequencyCountsSorted.forEach(element => {
        var makeKey = element.year;
         if(!frequencyGroupedYear[makeKey]) {
            frequencyGroupedYear[makeKey] = [];
         }
    
         frequencyGroupedYear[makeKey].push({
          id : makeKey,
          average_rating: element.average_rating,
          frequency: element.frequency
        });
       });
    
    var datasetLineChart = []
    for (let i = 0; i < Object.keys(frequencyGroupedYear).length; i++) {
        year = Object.keys(frequencyGroupedYear)[i]
        if (year >= 2015 && year <= 2019) {
            var nestedDict = {
                id: year,
                values: frequencyGroupedYear[year]
            }
            datasetLineChart.push(nestedDict)

          }
        }

    //Create rawArr for creating the line data in chart a
    var rawArrMovies = data.map(function(d,i) {
        if (d.year >= 2015 && d.year <=2020) {
        return {year:d.year,
                average_rating: parseInt(Math.floor(d.average_rating)),
                users_rated: parseInt(d.users_rated),
                name: d.name
        }

    }});
    var rawArrMoviesFiltered = rawArrMovies.filter(function(x) {
        return x !== undefined;
    });

    var moviesSorted = rawArrMoviesFiltered.sort(fieldSorter(['year','average_rating','-users_rated']))
    console.log(moviesSorted)

    function top5Movies(selectedPoint) {
        var moviesYearRating = []
        moviesSorted.forEach(function(d, i){
            if (d.year == selectedPoint.id && d.average_rating == selectedPoint.average_rating){
                movieDict = {
                    year:d.year,
                    name:d.name.substring(0,10),
                    users_rated:d.users_rated,
                    average_rating:d.average_rating
                }
                moviesYearRating.push(movieDict)
            }
        })
        //Now sort the result by user_rated
        var top5MoviesPreSort = moviesYearRating.sort(fieldSorter(['-users_rated'])).slice(0,5)
        var top5MoviesFinal = top5MoviesPreSort.sort(fieldSorter(['users_rated']))
        return top5MoviesPreSort
        
    }


//----------------------------SCALES----------------------------//
const xScale = d3.scaleTime().range([0,width]);
const yScale = d3.scaleLinear().rangeRound([height, 0]);
const colors = d3.scaleOrdinal(d3.schemeCategory10);

xScale.domain([0, d3.max(datasetLineChart, function(c) {
    return d3.max(c.values, function(d) {
        return parseInt(d.average_rating); });
        })
    ]);

yScale.domain([(0), d3.max(datasetLineChart, function(c) {
    return d3.max(c.values, function(d) {
        return d.frequency; });
        })
    ]);
  

//-----------------------------AXES-----------------------------//
const yaxis = d3.axisLeft()
    .scale(yScale);

 
const xaxis = d3.axisBottom()
    .tickFormat(d3.format("d"))
    .scale(xScale);

//----------------------------LINES-----------------------------//
const line = d3.line()
    .x(function(d) { return xScale(d.average_rating)+100; })
    .y(function(d) { return yScale(d.frequency)+50; });

let id = 0;
const ids = function () {
    return "line-"+id++;
}  

const point_ids = function () {
    return "point-"+id++;
}  

//-------------------------2. DRAWING---------------------------//
//-----------------------------AXES-----------------------------//

svg_linechart.append("g")
    .attr("id","line_chart_title")

svg_linechart.select("#line_chart_title")
    .append("text")
    .attr("transform", "translate(300,0)")
    .attr("y", 25)
    .attr("font-size", "20px")
    .text("Board games by Rating 2015-2019");

svg_linechart.append("g")
    .attr("id","credit")

svg_linechart.select("#credit")
    .append("text")
    .attr("y", height + 100 )
    .attr("x", width+100)
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("mpearl3");
    
svg_linechart.append("g")
    .attr("transform", "translate(100," + (height+50) + ")")
    .attr("id", "x-axis-lines")
    .call(xaxis);

svg_linechart.append("text")
    .attr("id", "x-axis label")
    .attr("transform", "translate(400,0)")
    .attr("y", height + 100)
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Rating");

svg_linechart.append("g")
    .attr("id", "y-axis-lines")
    .attr("transform", "translate(100,50)")
    .call(yaxis);

svg_linechart.append("text")
    .attr("id", "y-axis label")
    .attr("transform", "rotate(-90)")
    .attr("y", height - 150)
    .attr("x", -150)
    .attr("dy", "-5.1em")
    .attr("text-anchor", "end")
    .attr("stroke", "black")
    .text("Count");

//----------------------------LINES-----------------------------//

const frequencyLines = svg_linechart.append("g").attr("id","lines")
  .selectAll("lines")
  .data(datasetLineChart)
  .enter();

frequencyLines.append("path")
  .attr("fill", "none")
  .attr("d", function(d) { return line(d.values); })
  .style("stroke", function(d, i) {
    return colors(d.id);
});

frequencyPoints = svg_linechart.append("g")
    .attr("id","circles")
    

frequencyPoints
        .selectAll(".dot")
        .data(datasetLineChart)
        .enter()
        .append("g")
        .attr("id",point_ids)
        .selectAll("circle")
        .data(function(d){return d.values})
        .enter()
        .append("circle")
        .attr("cx", function(d) { 
            return xScale(d.average_rating)+100; 
        })      
        .attr("cy", function(d,i) { return yScale(d.frequency)+50; })    
        .attr("r", 4)
        .style("fill", function(d) {
            return colors(d.id);
        }
        )
        .on("mouseover", function(d) {
            d3.select(event.currentTarget).attr("r", "12");
            const svg_barchart = d3.select("body")
            .append("svg")
            .attr("id","barchart")
            .attr("width",global_width)
            .attr("height", global_height)    

            const topMoviesSelection = top5Movies(d)
            if (d.frequency > 0) {            
                //----------------------------SCALES----------------------------//
                const xScaleBar = d3.scaleLinear().range([0,width]);
                const yScaleBar = d3.scaleBand()
                .range([height, 0]);

                xScaleBar.domain([0, d3.max(topMoviesSelection,function(c) {
                    return c.users_rated;
                })]);

                var movieNames = [] 
                topMoviesSelection.forEach( 
                    function(d){
                        movieNames.push(d.name);
                    } 
                );

                yScaleBar.domain(movieNames);
                

                //-----------------------------AXES-----------------------------//
                const yaxisBar = d3.axisLeft()
                    .scale(yScaleBar);

                
                const xaxisBar = d3.axisBottom()
                    .tickFormat(d3.format("d"))
                    .scale(xScaleBar);
                //-----------------------------DRAWING SUB-PLOT-----------------------------//                
                
                svg_barchart.append("g")
                    .attr("id","bar_chart_title")
                    .data(topMoviesSelection)
                    .enter()                
                
                // const countLinesA = svg_a.select("#plot-a").append("g").attr("id","lines-a")
                //     .selectAll("lines")
                //     .data(countSlices)
                //     .enter();

                svg_barchart
                    .select("#bar_chart_title")
                    .append("text")
                    .datum(function(d) {
                        return {
                            year: d.year,
                            value: d.average_rating
                    }})                 
                    .attr("transform", "translate(300,0)")
                    .attr("y", 25)
                    .attr("font-size", "20px")
                    .text(function(d) { 
                        return 'Top 5 most rated games of '+d.year+' with rating '+d.value
                    });
            
                    
                svg_barchart.append("g")
                    .attr("transform", "translate(100," + (height+50) + ")")
                    .attr("id", "x-axis-bars")
                    .call(xaxisBar);

                svg_barchart.append("g")
                    .attr("id", "bar_x_axis_label");
                
                svg_barchart.select("#bar_x_axis_label")
                    .append("text")
                    .attr("transform", "translate(400,0)")
                    .attr("y", height +100)
                    .attr("text-anchor", "end")
                    .attr("stroke", "black")
                    .text("Number of Users");
                                
                svg_barchart.append("g")
                    .attr("id","y-axis-bars")
                    .attr("transform", "translate(100,50)")
                    .call(yaxisBar);

                svg_barchart.append("g")
                    .attr("id", "bar_y_axis_label");    
                
                svg_barchart.select("#bar_y_axis_label")
                    .append("text")
                    .attr("transform", "rotate(-90)")
                    .attr("y", height - 150)
                    .attr("x", -150)
                    .attr("dy", "-5.1em")
                    .attr("text-anchor", "end")
                    .attr("stroke", "black")
                    .text("Games");

                const bars = svg_barchart.append("g")
                    .attr("id","bars")
                    .selectAll("myRect")
                    .data(topMoviesSelection)
                    .enter();            

                bars.append("rect")
                    .attr("x", xScaleBar(0)+100 )
                    .attr("y", function(d) { return yScaleBar(d.name)+50; })
                    .attr("width", function(d) { return xScaleBar(d.users_rated); })
                    .attr("height", yScaleBar.bandwidth() )
                    .attr("fill", "#69b3a2");            
            }})
        .on("mouseout", function(d){
            d3.select(event.currentTarget).attr("r", 4);
            d3.select("#barchart").remove()
            // svg_barchart.selectAll(".text").remove()
            // svg_barchart.select("#x-axis-bars").remove()
            // svg_barchart.select("#y-axis-bars").remove()
            // svg_barchart.select("#bar-chart-title").remove()
            // svg_barchart.select("#bar_x_axis_label").remove()
            // svg_barchart.select("#bar_y_axis_label").remove()
            // svg_barchart.selectAll('#bars').remove()
            // svg_barchart.select("#bar_chart_title").remove()




                
        });


var legend_years = ["2015","2016","2017","2018","2019"]        

legend = svg_linechart.append("g")
    .attr("id","legend")

// Handmade legend
legend_years.forEach(function(d,i) {
    legend.append("circle").attr("cx",(width+40)+(1)*8).attr("cy",95+(i+1)*15).attr("r", 6).style("fill", colors(d))
    legend.append("text").attr("x", (width+50)+(1)*8).attr("y", 95+(i+1)*15).text(d).style("font-size", "15px")
}
)





});