//$("#cities").html('<option>city1</option><option>city2</option>');
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