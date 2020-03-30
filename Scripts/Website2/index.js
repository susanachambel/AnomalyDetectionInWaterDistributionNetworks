//$("#cities").html('<option>city1</option><option>city2</option>');~

$(function () {
    $('input[name="daterange"]').daterangepicker({
        locale: {
            format: 'DD/MM/YYYY'
        },
        opens: 'center',
        "minDate": "01/01/2017",
        "maxDate": "31/12/2017",
        "maxSpan": {
            "days": 7
        }
    }, function (start, end, label) {
        console.log("A new date selection was made: " + start.format('YYYY-MM-DD') + ' to ' + end.format('YYYY-MM-DD'));
    });
});

$('input[name="daterange"]').on('apply.daterangepicker', function (ev, picker) {
    console.log(picker.startDate.format('DD-MM-YYYY'));
    console.log(picker.endDate.format('DD-MM-YYYY'));
});



$('#empid').selectpicker('val', ['Tuples', 'Temporal']);

$('#empid').on('change', function () {
    var selected = []
    selected = $('#empid').val()
    console.log(selected); //Get the multiple values selected in an array
    console.log(selected.length); //Length of the array
});

//$('#empid').prop('disabled', true);

create_line_chart()
create_heat_map()

function create_line_chart() {

    var config = {
        responsive: true
    }

    var layout = {    
        xaxis: {

            rangeslider: {}
        }
    };

    var trace1 = 
        {
          x: ['2013-10-04 22:23:00', '2013-11-04 22:23:00', '2013-12-04 22:23:00'],
          y: [1, 3, 6],
          type: 'scatter'
        };

    var data = [trace1];

    Plotly.newPlot('line_chart', data, layout, config);
}

function create_heat_map() {

    var config = {
        responsive: true
    }

    var layout = {    
    };

    var data = [
        {
          z: [[1, null, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
          x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
          y: ['Morning', 'Afternoon', 'Evening'],
          colorscale: 'YIOrRd',
          reversescale: true,
          type: 'heatmap',
          hoverongaps: false
        }
      ];
      
      Plotly.newPlot('heat_map', data, layout, config);
}


