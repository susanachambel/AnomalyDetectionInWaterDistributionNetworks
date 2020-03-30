window.onload = function () {

    var forms = document.getElementsByClassName('needs-validation');
    var validation = Array.prototype.filter.call(forms, function (form) {
        form.addEventListener('submit', function (event) {

            validate_check_box("focus", 1);
            validate_check_box("sensor-group", 1);
            validate_check_box("pairwise-comparisons", 1);

            validate_selection("sensor-type", 1);
            validate_selection("sensor-name", 2);
            validate_selection("calendar", 1);
            validate_selection("correlation-type", 1);
            validate_selection("correlation", 1);

            if (form.checkValidity() === false) {
                event.preventDefault();
                event.stopPropagation();
                form.classList.add('was-validated');
            } else {
                submit();
                form.classList.remove('was-validated');
            };

        }, false);
    });

    document.getElementById("source").addEventListener("change", function () {
        switch (this.value) {
            case "infraquinta":
                activate_infraquinta();
                break;
            case "beja":
                activate_beja();
                break;
            case "barreiro":
                activate_barreiro();
                break;
            default:
                break;
        };
    });

    document.querySelectorAll("input[name=check-focus]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("focus", 1);
            };
        });
    });

    document.querySelectorAll("input[name=check-sensor-group]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("sensor-group", 1);
            };
        });
    });

    document.querySelectorAll("input[name=check-pairwise-comparisons]").forEach(box => {
        box.addEventListener("change", function () {
            if (document.getElementById("form-target").classList.contains('was-validated') === true) {
                validate_check_box("pairwise-comparisons", 1);
            };
        });
    });

    document.getElementById("sensor-type").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("sensor-type", 1);
        };
    });

    document.getElementById("sensor-name").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("sensor-name", 2);
        };
    });

    document.getElementById("calendar").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("calendar", 1);
        };
    });

    document.getElementById("correlation-type").addEventListener("change", function () {
        //var selected = [];
        //selected = $('#correlation-type').val();
        //console.log(selected); //Get the multiple values selected in an array
        //console.log(selected.length); //Length of the array
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("correlation-type", 1);
        };
    });

    document.getElementById("correlation").addEventListener("change", function () {
        if (document.getElementById("form-target").classList.contains('was-validated') === true) {
            validate_selection("correlation", 1);
        };
    });

    get_init_json();
    activate_infraquinta();
};

function get_init_json() {
    console.log("hello");
    var data = {
        name: "json"
    };
    $.post("receiver", data, receive_init_json);
};

function receive_init_json(data, status) {
    var data_json = JSON.parse(data);

    var infraquinta = data_json['infraquinta']
    var barreiro = data_json['barreiro']
    var beja = data_json['beja']

    console.log(data_json)
};


function submit() {
    var data = {
        name: "form"
    };

    deactivate_form();

    $.post("receiver", data, receive_data);
    event.preventDefault();
};

function receive_data(data, status) {

    var data_json = JSON.parse(data);

    lc_data = data_json["line_chart"];
    hm_data = data_json["heat_map"];

    create_line_chart(lc_data);
    create_heat_map();
    activate_form();
};

function validate_selection(field, length) {
    if ($('#' + field).val().length < length) {
        document.getElementById(field).setCustomValidity(' ');
        if (!document.getElementById("help-" + field)) {
            $("#div-" + field).append('<small id="help-' + field + '" class="text-muted">Select at least ' + length + ' </small>');
        };
    } else {
        document.getElementById(field).setCustomValidity('');
        if (document.getElementById("help-" + field)) {
            $("#help-" + field).remove();
        };
    }
};

function validate_check_box(field, length) {
    if ($("[name='check-" + field + "']:checked").length < length) {
        document.querySelectorAll("input[name=check-" + field + "]").forEach(box => {
            box.setCustomValidity(' ');
        });
        if (!document.getElementById("help-" + field)) {
            $("#div-" + field).append('<small id="help-' + field + '" class="text-muted">Check at least ' + length + ' </small>');
        };
    } else {
        document.querySelectorAll("input[name=check-" + field + "]").forEach(box => {
            box.setCustomValidity('');
        });
        if (document.getElementById("help-" + field)) {
            $("#help-" + field).remove();
        };
    };
};


//$("#cities").html('<option>city1</option><option>city2</option>');


/*
$('input[name="daterange"]').on('apply.daterangepicker', function (ev, picker) {
    console.log(picker.startDate.format('DD-MM-YYYY'));
    console.log(picker.endDate.format('DD-MM-YYYY'));
});
*/


//$('#correlation-type').prop('disabled', true);




function create_line_chart(lc_data) {
    var config = {
        responsive: true
    };

    var layout = {
        xaxis: {
            rangeslider: {}
        }
    };

    var data = [];

    for (sensor in lc_data) {
        console.log(sensor);
        var x = Object.keys(lc_data[sensor]);
        var y = Object.values(lc_data[sensor]);

        var trace = create_trace(sensor, x, y);
        data.push(trace);
    };

    Plotly.newPlot('line_chart', data, layout, config);
};

function create_trace(name, x, y) {
    var trace = {
        name: name,
        x: x,
        y: y,
        type: 'scatter'
    };
    return trace;
};


function create_heat_map() {

    var config = {
        responsive: true
    };

    var layout = {};

    var data = [{
        z: [
            [1, null, 30],
            [20, 1, 60],
            [30, 60, 1]
        ],
        x: ['Monday', 'Tuesday', 'Wednesday'],
        y: ['Morning', 'Afternoon', 'Evening'],
        colorscale: 'YIOrRd',
        reversescale: true,
        type: 'heatmap',
        hoverongaps: false
    }];

    Plotly.newPlot('heat_map', data, layout, config);
};

function activate_infraquinta() {

    $('#source').selectpicker('val', 'infraquinta');

    $('#correlation-type').selectpicker('refresh');
    $('#correlation-type').selectpicker('selectAll');
    $('#correlation').selectpicker('refresh');
    $('#correlation').selectpicker('selectAll');
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-type').selectpicker('selectAll');
    $('#sensor-name').selectpicker('refresh');
    $('#sensor-name').selectpicker('selectAll');
    $('#calendar').selectpicker('refresh');
    $('#calendar').selectpicker('selectAll');
    
    $('#granularity').selectpicker('val', 'hours');

    $("[name='check-focus'").prop( "checked", true );
    $("[name='check-focus'").prop( "disabled", false );
    $("[name='check-sensor-group'").prop( "checked", true );
    $("[name='check-sensor-group'").prop( "disabled", false );
    $("[name='check-pairwise-comparisons'").prop( "checked", true );
    $("[name='check-pca'").prop( "checked", true );

    document.getElementById("check-default").checked = true;

    document.getElementById("granularity-value").value = 1;
    create_date_range_picker("01/06/2017", "08/06/2017", "01/01/2017", "31/12/2017", 7);
};

function activate_barreiro() {

    console.log("barreiro");
    //$("#div-source").html('<div class="form-check form-check-inline"><input class="form-check-input" type="checkbox" id="inlineCheckbox1" value="real"><label class="form-check-label" for="inlineCheckbox1">Real</label></div>');
}

function activate_beja() {

    console.log("beja");

};

function create_date_range_picker(startDate, endDate, minDate, maxDate, maxSpan){
    $(function () {
        $('input[name="daterange"]').daterangepicker({
            locale: {
                format: 'DD/MM/YYYY'
            },
            opens: 'center',
            "startDate": startDate,
            "endDate": endDate,
            "minDate": minDate,
            "maxDate": maxDate,
            "maxSpan": {
                "days": maxSpan
            }
        }, function (start, end, label) {
            console.log("A new date selection was made: " + start.format('YYYY-MM-DD') + ' to ' + end.format('YYYY-MM-DD'));
        });
    });
};




function deactivate_form() {
    $('#run-btn').prop("disabled", true);
    $("#run-btn").html(
        `<span id="btn-spinner" class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Running...`
    );

    $('#source').prop("disabled", true);
    $('#source').selectpicker('refresh');
    $('#form-target input[type=checkbox]').prop("disabled", true);
    $('#sensor-type').prop("disabled", true);
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-name').prop("disabled", true);
    $('#sensor-name').selectpicker('refresh');
    $('#calendar').prop("disabled", true);
    $('#calendar').selectpicker('refresh');
    $('#form-target input[type=text]').prop("disabled", true);
    $('#granularity').prop("disabled", true);
    $('#granularity').selectpicker('refresh');
    $('#form-target input[type=number]').prop("disabled", true);

    $('#form-target input[type=radio]').prop("disabled", true);
    $('#correlation-type').prop('disabled', true);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation').prop('disabled', true);
    $('#correlation').selectpicker('refresh');
};

function activate_form() {
    $('#source').prop("disabled", false);
    $('#source').selectpicker('refresh');
    $('#form-target input[type=checkbox]').prop("disabled", false);
    $('#sensor-type').prop("disabled", false);
    $('#sensor-type').selectpicker('refresh');
    $('#sensor-name').prop("disabled", false);
    $('#sensor-name').selectpicker('refresh');
    $('#calendar').prop("disabled", false);
    $('#calendar').selectpicker('refresh');
    $('#form-target input[type=text]').prop("disabled", false);
    $('#granularity').prop("disabled", false);
    $('#granularity').selectpicker('refresh');
    $('#form-target input[type=number]').prop("disabled", false);

    $('#form-target input[type=radio]').prop("disabled", false);
    $('#correlation-type').prop('disabled', false);
    $('#correlation-type').selectpicker('refresh');
    $('#correlation').prop('disabled', false);
    $('#correlation').selectpicker('refresh');

    $("#btn-spinner").remove();
    $("#run-btn").html(
        `Run Query`
    );
    $("#run-btn").prop("disabled", false);
};